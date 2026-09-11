"""P5 Step6: KV捕获->序列化->恢复继续, 验证与单机直推等价."""
import sys, time, io
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from transport.adapter import serialize_kv, deserialize_kv
from telemetry.logger import log
from eval.model_loader import load_causal, model_id, model_device

MODEL = model_id()
PROMPT = "The capital of France is"
K, M = 8, 8  # A生成K个, B续M个

tok, model = load_causal(MODEL)
DEV = model_device(model)

def step_forward(input_id, kv=None):
    with torch.no_grad():
        out = model(input_ids=input_id, past_key_values=kv, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return nxt, out.past_key_values

# A侧: 从prompt逐步生成K步
ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
kv = None; t0 = time.time()
with torch.no_grad():
    out = model(input_ids=ids, use_cache=True); kv = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    seq = torch.cat([ids, nxt], dim=1)
    for _ in range(K - 1):
        nxt, kv = step_forward(nxt, kv); seq = torch.cat([seq, nxt], dim=1)
tA = round(time.time() - t0, 2)
b = serialize_kv(kv)
print(f"A done: seq={seq.shape}, KV bytes={len(b)}, t={tA}s")

# 模拟传输
t0 = time.time(); kv2 = deserialize_kv(b); tTx = round(time.time() - t0, 3)

# B侧: 用恢复的KV续生成M步
seqB = seq.clone(); last = seq[:, -1:]; t0 = time.time()
for _ in range(M):
    last, kv2 = step_forward(last, kv2); seqB = torch.cat([seqB, last], dim=1)
tB = round(time.time() - t0, 2)
txtB = tok.decode(seqB[0].cpu(), skip_special_tokens=True)

# 基线: 单机直推K+M步
ids0 = tok(PROMPT, return_tensors="pt")["input_ids"].to(DEV)
with torch.no_grad():
    base = model.generate(**{"input_ids": ids0}, max_new_tokens=K+M, do_sample=False)
txt0 = tok.decode(base[0].cpu(), skip_special_tokens=True)

print("KV续写:", txtB)
print("单机直推:", txt0)
equiv = (seqB.shape == base.shape) and torch.equal(seqB, base)
print("EQUIV:", equiv)
log("kv_equiv", equiv=equiv, kv_bytes=len(b), tA=tA, tB=tB, layers=len(kv), seq=list(seq.shape))
assert equiv, "KV恢复续写与直推不一致"
print("Step6 等价性通过")
