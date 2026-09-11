"""Task3 跨机讨论器: A初答→B批评→A修正, 全程走TCP, 混合模式(A→B TOKEN, B→A KV).
用法: 先启动 B: python discuss.py --role b --port 5102
      再启动 A: python discuss.py --role a --host 127.0.0.1 --port 5102 --task math --n 5
      服务器上把 --host 改为B机IP即可.
"""
import sys, time, argparse
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transport.adapter import TcpTransport, LocalTransport, serialize_kv, deserialize_kv
from transport.link import send_text, send_kv, recv_payload
from telemetry.logger import log
from eval.check_answer import is_correct
from eval.dataset import SELFMADE
from eval.maze import load_mazes, prompt_for, extract_moves, check_path, bfs_path

MODEL = "Qwen/Qwen3-0.6B"
MAX_N = 128
MAX_ROUNDS = 2

def p_a1_math(q): return f"Question: {q}\nSolve it briefly. End your answer with the final number only.\nAnswer:"
def p_b_math(q, a1): return f"Question: {q}\nFirst answer: {a1}\nCriticize briefly, then give the corrected final number at the very end.\nCritic:"
def p_a2_math(q, a1, c): return f"Question: {q}\nFirst: {a1}\nCritic: {c}\nGive the final number only.\nFinal:"
def p_b_maze(grid, p1): return f"Maze:\n{grid}\nProposed path: {p1}\nCheck each step for wall collisions. Give the corrected full move sequence at the very end.\nCritic:"
def p_a2_maze(grid, p1, c): return f"Maze:\n{grid}\nFirst path: {p1}\nCritic: {c}\nGive the final move sequence only.\nFinal:"
def p_v_math(q, draft): return f"Question: {q}\nCritic draft: {draft}\nVerify the reasoning and give the final corrected number at the very end.\nVerify:"
def p_v_maze(grid, draft): return f"Maze:\n{grid}\nCritic draft: {draft}\nVerify each step for wall collisions and give the final move sequence at the very end.\nVerify:"

tok = None; model = None
def load_model():
    global tok, model
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True)
    model.eval()

def gen_text(prompt, n=MAX_N):
    ids = tok(prompt, return_tensors="pt")["input_ids"]; t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False)
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return txt, round(time.time() - t0, 2)

def gen_with_kv(prompt, n=MAX_N):
    ids = tok(prompt, return_tensors="pt")["input_ids"]; t0 = time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    seq = out.sequences[0]
    with torch.no_grad():
        kv = model(seq.unsqueeze(0), use_cache=True).past_key_values
    txt = tok.decode(seq[ids.shape[1]:], skip_special_tokens=True).strip()
    return txt, kv, seq, round(time.time() - t0, 2)

def judge(task, item, a2):
    if task == "math":
        s, _ = is_correct(a2, item["answer"])
        return s
    mv = extract_moves(a2)
    ok, _, _ = check_path(item["grid_obj"], mv)
    return ok

def run_b(tp, agents=("critic",)):
    qid = -1
    while True:
        try:
            mtype, data, meta = recv_payload(tp, timeout=60)
        except Exception:
            time.sleep(1)  # 等待对端中, 避免空转烧CPU; 工作台停止按钮直接杀进程
            continue
        if meta.get("task") == "END":
            print("B: discussion ended.", flush=True); break
        assert mtype == "TOKEN"
        task, q, a1, rnd = meta["task"], meta["question"], data.decode("utf-8"), meta["round"]
        grid = meta.get("grid", "")
        if rnd == 1:
            qid += 1
        log("discuss_b_msg", task=task, qid=qid, round=rnd, question=q, proposal=a1,
            grid=grid if task == "maze" else "")
        text, kv, seq, dt_total = "", None, None, 0.0
        prev = a1
        chain = []
        for ag in agents:
            if ag == "critic":
                prompt = p_b_math(q, prev) if task == "math" else p_b_maze(grid, prev)
            else:  # verifier 及后续: 在上一智能体输出上继续校验
                prompt = p_v_math(q, prev) if task == "math" else p_v_maze(grid, prev)
            text, kv, seq, dt = gen_with_kv(prompt)
            dt_total += dt; prev = text; chain.append(ag)
        raw = serialize_kv(kv)
        send_kv(tp, raw, hop="B->A", mode="KV", n_bytes=len(raw), ids=seq.tolist(),
                infer_s=round(dt_total, 2), round=rnd, chain="+".join(chain))
        log("discuss_b", task=task, qid=qid, round=rnd, infer_s=round(dt_total, 2),
            kv_bytes=len(raw), chain="+".join(chain), critic=text)
        print(f"B: round{rnd} chain={'+'.join(chain)} replied ({len(raw)}B KV).", flush=True)

def run_a(tp, task, n):
    items = ([{"q": q, "answer": a} for q, a, _, _ in SELFMADE[:n]] if task == "math"
             else [{"q": m["grid"], "grid_obj": __import__("eval.maze", fromlist=["gen_maze"]).gen_maze(m["size"], seed=int(m["id"].split("-")[1]) + (1000 if m["size"] == 5 else 2000))} for m in load_mazes()[:n]])
    # 注意: maze grid_obj 用同seed重建, 与题集一致
    wins, rounds_used = 0, 0
    for qi, item in enumerate(items):
        q = item["q"]; proposal = None; ok = False
        for rnd in range(1, MAX_ROUNDS + 1):
            if rnd == 1:
                if task == "math":
                    proposal, t1 = gen_text(p_a1_math(q))
                else:
                    proposal, t1 = gen_text(prompt_for(item["grid_obj"]))
            t0 = time.time()
            send_text(tp, proposal, task=task, question=q, round=rnd, hop="A->B", mode="TOKEN",
                      grid=item["q"] if task == "maze" else "")
            mtype, kv_raw, meta = recv_payload(tp, timeout=600)
            net = round(time.time() - t0, 2)
            assert mtype == "KV"
            kv2 = deserialize_kv(kv_raw)  # 恢复即证明到达
            _ = kv2
            prompt_ids = tok((p_b_math(q, proposal) if task == "math" else p_b_maze(item["q"], proposal)), return_tensors="pt")["input_ids"]
            critic = tok.decode(torch.tensor([meta["ids"]])[:, prompt_ids.shape[1]:][0], skip_special_tokens=True).strip()
            if task == "math":
                a2, t2 = gen_text(p_a2_math(q, proposal, critic))
            else:
                a2, t2 = gen_text(p_a2_maze(item["q"], proposal, critic))
            ok = judge(task, item, a2)
            rounds_used += 1
            log("discuss_a", task=task, qid=qi, round=rnd, ok=ok, net_s=net,
                kv_bytes=meta.get("n_bytes", 0), infer_s=round(t1 + t2, 2) if rnd == 1 else round(t2, 2),
                question=q, proposal=proposal, critic=critic, final=a2)
            print(f"A: q{qi} round{rnd} ok={ok} net={net}s kv={meta.get('n_bytes',0)}B", flush=True)
            if ok: break
            proposal = a2
        wins += ok
    send_text(tp, "END", task="END")
    print(f"A: done {wins}/{len(items)} success.", flush=True)
    log("discuss_summary", task=task, wins=wins, total=len(items), rounds=rounds_used)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["a", "b"], required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--task", choices=["math", "maze"], default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--config", default=str(Path(ROOT) / "config.yaml"))
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    host = args.host or cfg["b"]["host"]
    port = args.port or cfg["b"]["port"]
    task = args.task or cfg["discuss"]["task"]
    n = args.n or cfg["discuss"]["n"]
    global MAX_ROUNDS
    MAX_ROUNDS = cfg["discuss"].get("max_rounds", 2)
    agents = tuple(cfg["b"].get("agents", ["critic"]))
    load_model()
    tp = None
    try:
        if args.role == "b":
            print(f"B listening... agents={'+'.join(agents)}", flush=True)
            tp = TcpTransport(host=host, port=port, is_server=True)
            run_b(tp, agents)
        else:
            for i in range(30):
                try:
                    tp = TcpTransport(host=host, port=port, is_server=False); break
                except ConnectionRefusedError:
                    print(f"wait server...{i}", flush=True); time.sleep(10)
            if tp is None: raise ConnectionError("server not ready")
            run_a(tp, task, n)
    finally:
        if tp is not None:
            tp.close()

if __name__ == "__main__":
    main()
