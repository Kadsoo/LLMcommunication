"""P3 server = Agent B(批评). 先启动: python server_b.py"""
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transport.adapter import TcpTransport, Message, TOKEN
from telemetry.logger import log

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, trust_remote_code=True)
model.eval()

tp = TcpTransport(port=5101, is_server=True)
print("B listening...", flush=True)
m = tp.recv(timeout=120)
q = json_q = m["meta"]["question"]; a1 = Message.data(m).decode()
t0 = time.time()
inputs = tok(f"Question: {q}\nProposed answer: {a1}\nCriticize in one paragraph:\nCritic:", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
critic = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
dt = round(time.time() - t0, 2)
tp.send(Message.make(TOKEN, critic.encode(), {"infer_s": dt}))
log("tcp_b", role="B", infer_s=dt, payload_in=len(Message.data(m)), payload_out=len(critic.encode()))
print("B replied.", flush=True); tp.close()
