"""P6 Step7: 同题对比 TOKEN vs KV (成功率/时延/传输量)."""
import sys, time
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transport.adapter import serialize_kv, deserialize_kv
from telemetry.logger import log

MODEL = "Qwen/Qwen3-0.6B"
QS = [
    ("Janet 3 apples +5, total?", "8"),
    ("60km/h *3h = ?", "180"),
    ("30 bees leave, half return, returned?", "15"),
    ("10-4 = ?", "6"),
    ("2+3*4 = ?", "14"),
]
K, M = 16, 48

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

rows=[]
for q, exp in QS:
    # TOKEN模式: A全量生成后传文本
    t0=time.time()
    a_full, ta, na = gen_full(f"Question: {q}\nAnswer:", 64)
    payload_tok = len(a_full.encode())
    ok_t = exp in a_full
    t_tok = round(time.time()-t0,2)
    # KV模式: A生成K步传KV, B续M步
    t0=time.time()
    seq, kv, tA = gen_prefix_kv(f"Question: {q}\nAnswer:", K)
    b = serialize_kv(kv); kv2 = deserialize_kv(b)
    txt, tB = cont_from_kv(seq, kv2, M)
    ok_k = exp in txt
    t_kv = round(time.time()-t0,2)
    rows.append({"q":q,"tok_ok":ok_t,"tok_s":t_tok,"tok_bytes":payload_tok,"tok_n":na,
                 "kv_ok":ok_k,"kv_s":t_kv,"kv_bytes":len(b)})
    d = dict(rows[-1]); d.pop("q"); log("compare", question=q, **d)
    print(f"{q} | TOK ok={ok_t} {t_tok}s {payload_tok}B | KV ok={ok_k} {t_kv}s {len(b)}B", flush=True)

# 结果表
s_t = sum(r["tok_ok"] for r in rows); s_k = sum(r["kv_ok"] for r in rows)
md = ["| 模式 | 成功率 | 平均时延 | 平均传输量 |", "|---|---|---|---|",
 f"| TOKEN | {s_t}/5 | {sum(r['tok_s'] for r in rows)/5:.2f}s | {sum(r['tok_bytes'] for r in rows)/5:.0f}B |",
 f"| KV | {s_k}/5 | {sum(r['kv_s'] for r in rows)/5:.2f}s | {sum(r['kv_bytes'] for r in rows)/5:.0f}B |",
 "", "注: CPU实测, KV负载大但省Prefill; 上GPU/服务器后时延比会更明显。"]
Path(ROOT,"eval","results_table.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
