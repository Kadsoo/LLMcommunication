"""TaskB 选型对比: critic风格 strict(严格挑错) vs mild(温和复核), 固定A→B→A一轮 x50题.
复用 dialogue_modes 的solver/revise与gen. 支持 --limit N 冒烟与 --resume 续跑.
用法: python eval/role_ablation.py [--limit 5] [--resume]
"""
import sys, time, argparse, json
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
from telemetry.logger import log
from eval.dataset import load_dataset
from eval.check_answer import is_correct
from eval.dialogue_modes import gen, SOLVER_TMPL, REVISE_TMPL

CRITICS = {
    "strict": ("Question: {q}\nProposed answer: {a1}\nAct as a strict reviewer: "
               "check every arithmetic step for errors, point out the first error you find. "
               "If fully correct, say CORRECT.\nCritic:"),
    "mild": ("Question: {q}\nProposed answer: {a1}\nAct as a gentle reviewer: "
             "only point out clear mistakes, do not nitpick. "
             "If correct, say CORRECT.\nCritic:"),
}


def run_once(q, critic_tmpl):
    t0 = time.time()
    a1, _ = gen(SOLVER_TMPL.format(q=q))
    b, _ = gen(critic_tmpl.format(q=q, a1=a1))
    a2, _ = gen(REVISE_TMPL.format(q=q, a1=a1, b=b))
    return a2, round(time.time() - t0, 2)


def load_done():
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
        if r.get("event") == "ablation" and "variant" in r:
            done[(r["variant"], r.get("question", ""))] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    ds = load_dataset()
    if args.limit > 0:
        ds = ds[:args.limit]
    done = load_done() if args.resume else {}
    for variant, tmpl in CRITICS.items():
        for i, d in enumerate(ds, 1):
            q, ans, cat = d["question"], d["answer"], d["category"]
            if (variant, q) in done:
                print(f"[{variant:6s} {i:02d}/{len(ds)}] skip (logged)", flush=True)
                continue
            a2, dt = run_once(q, tmpl)
            st, lo = is_correct(a2, ans)
            log("ablation", variant=variant, question=q, category=cat,
                strict=st, loose=lo, dt=dt)
            print(f"[{variant:6s} {i:02d}/{len(ds)}] strict={st} loose={lo} {dt:5.1f}s | {q[:44]}", flush=True)

    done = load_done()
    allrows = {}
    for variant in CRITICS:
        rows = []
        for d in ds:
            r = done.get((variant, d["question"]))
            if r is None:
                continue
            rows.append({"cat": d["category"], "strict": bool(r["strict"]),
                         "loose": bool(r["loose"]), "dt": r["dt"]})
        allrows[variant] = rows
    n = len(ds)
    missing = {v: n - len(allrows[v]) for v in CRITICS if len(allrows[v]) < n}
    if missing:
        print(f"WARN 缺数据(先--resume补跑): {missing}", flush=True)
    md = [f"## Critic选型对比 ({n}题, A→B→A一轮)",
          "| critic风格 | 严格通过 | 宽松通过 | 平均时延 |",
          "|---|---|---|---|"]
    for variant in CRITICS:
        rows = allrows[variant]; m = len(rows) or 1
        s = sum(r["strict"] for r in rows); lo = sum(r["loose"] for r in rows)
        t = sum(r["dt"] for r in rows) / m
        md.append(f"| {variant} | {s}/{len(rows)} ({s/m*100:.0f}%) | {lo}/{len(rows)} | {t:.1f}s |")
    md.append("")
    cats = sorted({d["category"] for d in ds})
    md.append("## 分题型严格通过率")
    md.append("| 题型 | strict(严格挑错) | mild(温和复核) |")
    md.append("|---|---|---|")
    for c in cats:
        cells = []
        for variant in CRITICS:
            sub = [r for r in allrows[variant] if r["cat"] == c]
            s = sum(r["strict"] for r in sub)
            cells.append(f"{s}/{len(sub)}" if sub else "-")
        md.append(f"| {c} | {' | '.join(cells)} |")
    Path(ROOT, "eval", "role_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
