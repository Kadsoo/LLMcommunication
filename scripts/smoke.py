"""冒烟测试: 用统一loader加载模型并生成一句话, 验证环境可用.
LLM_MODEL 环境变量切模型(默认 Qwen/Qwen3-0.6B).
"""
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from eval.model_loader import load_causal, model_id, model_device

tok, model = load_causal()
DEV = model_device(model)
ids = tok("1+1=", return_tensors="pt")["input_ids"].to(DEV)
with torch.no_grad():
    out = model.generate(input_ids=ids, max_new_tokens=16, do_sample=False)
print("SMOKE:", tok.decode(out[0].cpu(), skip_special_tokens=True).strip()[:80])
print(f"SMOKE PASS model={model_id()} device={DEV}")
