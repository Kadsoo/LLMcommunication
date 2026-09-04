"""P3 client = Agent A(解答+修正). 后启动: python client_a.py"""
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transport.adapter import TcpTransport, Message, TOKEN
from telemetry.logger import log

Q = "Debra sees 30 bees leave, half that many return. How many returned?"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, trust_remote_code=True)
model.eval()

def gen(prompt, n=64):
    inputs = tok(prompt, return_tensors="pt"); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip(), round(time.time()-t0, 2)

a1, t1 = gen(f"Question: {Q}\nSolve briefly:\nAnswer:")
tp = None
for i in range(30):
    try:
        tp = TcpTransport(port=5101, is_server=False)
        break
    except ConnectionRefusedError:
        print(f"wait server...{i}", flush=True); time.sleep(10)
if tp is None: raise ConnectionError("server not ready")
t0 = time.time()
tp.send(Message.make(TOKEN, a1.encode(), {"question": Q}))
m = tp.recv(timeout=120); net = round(time.time()-t0, 2)
critic = Message.data(m).decode()
a2, t3 = gen(f"Question: {Q}\nFirst: {a1}\nCritic: {critic}\nFinal number:\nFinal:")
log("tcp_a", infer_s=round(t1+t3,2), net_s=net, payload=len(a1.encode()))
print(f"A1: {a1[:300]}\nB: {critic[:300]}\nA2: {a2[:300]}\nnet={net}s"); tp.close()
