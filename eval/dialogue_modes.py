"""TaskA 对话模式对比: single直答 / fixed固定一轮A→B→A / dynamic动态停止.
复用 dataset(50题) + check_answer(严格/宽松) + model_loader.
用法: python eval/dialogue_modes.py [--limit N]  (先 --limit 5 冒烟)
"""
import sys, time, argparse, json
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from transport.adapter import serialize_kv, deserialize_kv  # noqa: F401 (协议一致, 本实验走TOKEN)
from telemetry.logger import log
from eval.dataset import load_dataset
from eval.check_answer import is_correct
from eval.model_loader import load_causal, model_id, model_device

MODEL = model_id()
MAX_N = 128
MAX_ROUNDS = 2

SOLVER_TMPL = "Question: {q}\nSolve it briefly. End your answer with the final number only.\nAnswer:"
CRITIC_TMPL = ("Question: {q}\nProposed answer: {a1}\nCriticize errors in one short paragraph. "
               "If correct, say CORRECT.\nCritic:")
REVISE_TMPL = ("Question: {q}\nFirst answer: {a1}\nCritic: {b}\n"
               "Give corrected final answer with number.\nFinal:")

tok, model = load_causal(MODEL)
DEV = model_device(model)


def gen(prompt, n=MAX_N):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV); t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False)
    txt = tok.decode(out[0].cpu()[ids.shape[1]:], skip_special_tokens=True).strip()
    return txt, round(time.time() - t0, 2)


def run_single(q):
    t0 = time.time()
    a, _ = gen(SOLVER_TMPL.format(q=q))
    return a, 1, round(time.time() - t0, 2)


def run_fixed(q):
    t0 = time.time()
    a1, _ = gen(SOLVER_TMPL.format(q=q))
    b, _ = gen(CRITIC_TMPL.format(q=q, a1=a1))
    a2, _ = gen(REVISE_TMPL.format(q=q, a1=a1, b=b))
    return a2, 3, round(time.time() - t0, 2)


def run_dynamic(q):
    t0 = time.time(); gens = 0
    a1, _ = gen(SOLVER_TMPL.format(q=q)); gens += 1
    b, _ = gen(CRITIC_TMPL.format(q=q, a1=a1)); gens += 1
    if "correct" in b.lower():
        return a1, gens, round(time.time() - t0, 2), 1  # B认可初答, 免修正
    a2, _ = gen(REVISE_TMPL.format(q=q, a1=a1, b=b)); gens += 1
    b2, _ = gen(CRITIC_TMPL.format(q=q, a1=a2)); gens += 1
    if "correct" in b2.lower():
        return a2, gens, round(time.time() - t0, 2), 2
    a3, _ = gen(REVISE_TMPL.format(q=q, a1=a2, b=b2)); gens += 1
    return a3, gens, round(time.time() - t0, 2), 2


def load_done():
    """已落盘的 (mode, question) -> 最新记录, 供断点续跑与制表."""
    done = {}
    p = Path(ROOT) / "telemetry" / "runs.jsonl"
    if not p.exists():
        return done
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "dialogue" and "mode" in r:
            done[(r["mode"], r.get("question", ""))] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="跳过telemetry中已有的(mode, question)")
    args = ap.parse_args()
    ds = load_dataset()
    if args.limit > 0:
        ds = ds[:args.limit]
    done = load_done() if args.resume else {}
    modes = {"single": run_single, "fixed": run_fixed, "dynamic": run_dynamic}
    for mode, fn in modes.items():
        for i, d in enumerate(ds, 1):
            q, ans, cat = d["question"], d["answer"], d["category"]
            if (mode, q) in done:
                print(f"[{mode:7s} {i:02d}/{len(ds)}] skip (logged)", flush=True)
                continue
            r = fn(q)
            final, ngen, dt = r[0], r[1], r[2]
            rounds = r[3] if mode == "dynamic" else 1
            st, lo = is_correct(final, ans)
            log("dialogue", mode=mode, question=q, category=cat, strict=st, loose=lo,
                ngen=ngen, dt=dt, rounds=rounds)
            print(f"[{mode:7s} {i:02d}/{len(ds)}] strict={st} loose={lo} gens={ngen} {dt:5.1f}s | {q[:44]}", flush=True)

    # 制表: 全部从telemetry取最新(支持续跑拼表)
    done = load_done()
    allrows = {}
    for mode in modes:
        rows = []
        for d in ds:
            r = done.get((mode, d["question"]))
            if r is None:
                continue
            rows.append({"cat": d["category"], "strict": bool(r["strict"]), "loose": bool(r["loose"]),
                         "ngen": r["ngen"], "dt": r["dt"], "rounds": r.get("rounds", 1)})
        allrows[mode] = rows
    n = len(ds)
    missing = {m: n - len(allrows[m]) for m in modes if len(allrows[m]) < n}
    if missing:
        print(f"WARN 缺数据(先--resume补跑): {missing}", flush=True)
    md = [f"## 对话模式对比 ({n}题, {MODEL})",
          "| 模式 | 严格通过 | 宽松通过 | 平均生成次数 | 平均时延 |",
          "|---|---|---|---|---|"]
    for mode in modes:
        rows = allrows[mode]; m = len(rows) or 1
        s = sum(r["strict"] for r in rows); lo = sum(r["loose"] for r in rows)
        g = sum(r["ngen"] for r in rows) / m; t = sum(r["dt"] for r in rows) / m
        extra = ""
        if mode == "dynamic":
            extra = f" | 平均轮次 {sum(r['rounds'] for r in rows) / m:.2f}"
        md.append(f"| {mode} | {s}/{len(rows)} ({s/m*100:.0f}%) | {lo}/{len(rows)} | {g:.1f} | {t:.1f}s |{extra}")
    md.append("")
    cats = sorted({d["category"] for d in ds})
    md.append("## 分题型严格通过率")
    md.append("| 题型 | single | fixed | dynamic |")
    md.append("|---|---|---|---|")
    for c in cats:
        cells = []
        for mode in modes:
            sub = [r for r in allrows[mode] if r["cat"] == c]
            s = sum(r["strict"] for r in sub)
            cells.append(f"{s}/{len(sub)}")
        md.append(f"| {c} | {' | '.join(cells)} |")
    Path(ROOT, "eval", "dialogue_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
