"""P6 Step7: 50题同题对比 TOKEN vs KV (成功率strict/loose, 时延, 传输量, 分题型)."""
import sys, time
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transport.adapter import serialize_kv, deserialize_kv
from telemetry.logger import log
from eval.dataset import load_dataset
from eval.check_answer import is_correct

MODEL = "Qwen/Qwen3-0.6B"
K, M = 16, 48
MAX_N = 96

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True)
model.eval()

def gen_full(prompt, n):
    ids = tok(prompt, return_tensors="pt")["input_ids"]; t0=time.time()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), round(time.time()-t0,2), int(out[0].shape[0]-ids.shape[1])

def gen_prefix_kv(prompt, k):
    ids = tok(prompt, return_tensors="pt")["input_ids"]; t0=time.time()
    with torch.no_grad():
        out = model(ids, use_cache=True); kv = out.past_key_values
        nxt = out.logits[:,-1,:].argmax(-1, keepdim=True); seq = torch.cat([ids,nxt],1)
        for _ in range(k-1):
            o = model(nxt, past_key_values=kv, use_cache=True)
            nxt = o.logits[:,-1,:].argmax(-1, keepdim=True); kv = o.past_key_values
            seq = torch.cat([seq,nxt],1)
    return seq, kv, round(time.time()-t0,2)

def cont_from_kv(seq, kv, m):
    last = seq[:,-1:]; t0=time.time()
    with torch.no_grad():
        for _ in range(m):
            o = model(last, past_key_values=kv, use_cache=True)
            last = o.logits[:,-1,:].argmax(-1, keepdim=True); kv=o.past_key_values
            seq = torch.cat([seq,last],1)
    return tok.decode(seq[0], skip_special_tokens=True), round(time.time()-t0,2)

def main():
    ds = load_dataset()
    rows = []
    for i, d in enumerate(ds, 1):
        q, ans, cat = d["question"], d["answer"], d["category"]
        # TOKEN模式
        t0=time.time()
        a_full, ta, na = gen_full(f"Question: {q}\nAnswer:", MAX_N)
        payload_tok = len(a_full.encode())
        st, lo = is_correct(a_full, ans)
        t_tok = round(time.time()-t0,2)
        # KV模式
        t0=time.time()
        seq, kv, tA = gen_prefix_kv(f"Question: {q}\nAnswer:", K)
        b = serialize_kv(kv); kv2 = deserialize_kv(b)
        txt, tB = cont_from_kv(seq, kv2, M)
        sk, lk = is_correct(txt, ans)
        t_kv = round(time.time()-t0,2)
        r = {"q": q, "ans": ans, "cat": cat,
             "tok_strict": st, "tok_loose": lo, "tok_s": t_tok, "tok_bytes": payload_tok,
             "kv_strict": sk, "kv_loose": lk, "kv_s": t_kv, "kv_bytes": len(b)}
        rows.append(r)
        log("compare", question=q, answer=ans, category=cat, **{k: v for k, v in r.items() if k not in ("q", "ans", "cat")})
        print(f"[{i:02d}/{len(ds)}] {cat:12s} TOK {st}/{lo} {t_tok:5.1f}s {payload_tok:6d}B | KV {sk}/{lk} {t_kv:5.1f}s {len(b):8d}B | {q[:52]}", flush=True)

    n = len(rows)
    s_t = sum(r["tok_strict"] for r in rows); l_t = sum(r["tok_loose"] for r in rows)
    s_k = sum(r["kv_strict"] for r in rows); l_k = sum(r["kv_loose"] for r in rows)
    at = sum(r["tok_s"] for r in rows) / n; ak = sum(r["kv_s"] for r in rows) / n
    bt = sum(r["tok_bytes"] for r in rows) / n; bk = sum(r["kv_bytes"] for r in rows) / n
    md = ["## 总体 (50题, Qwen3-0.6B 本机CPU)", "| 模式 | 严格通过 | 宽松通过 | 平均时延 | 平均传输量 |",
          "|---|---|---|---|---|",
          f"| TOKEN | {s_t}/{n} ({s_t/n*100:.0f}%) | {l_t}/{n} | {at:.2f}s | {bt:.0f}B |",
          f"| KV | {s_k}/{n} ({s_k/n*100:.0f}%) | {l_k}/{n} | {ak:.2f}s | {bk:.0f}B |", ""]
    cats = sorted({r["cat"] for r in rows})
    md.append("## 分题型 (严格通过率)")
    md.append("| 题型 | 题数 | TOKEN | KV |")
    md.append("|---|---|---|---|")
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        ts = sum(r["tok_strict"] for r in sub); ks = sum(r["kv_strict"] for r in sub)
        md.append(f"| {c} | {len(sub)} | {ts}/{len(sub)} | {ks}/{len(sub)} |")
    Path(ROOT, "eval", "results_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

if __name__ == "__main__":
    main()