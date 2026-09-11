"""统一模型加载: 环境变量切模型 + 自动设备/精度/量化.

环境变量:
  LLM_MODEL  模型ID, 默认 Qwen/Qwen3-0.6B(本地行为不变)
  LLM_4BIT   设为 1 强制 4bit 量化(需 bitsandbytes)
自动策略(GPU):
  显存 >= 16GB -> float16 整卡加载
  显存 < 16GB  -> 4bit 量化加载
CPU: float32(与原来一致).
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
FOURBIT_VRAM = 16 * 1024 ** 3


def model_id():
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def _vram_bytes():
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory
        except Exception:
            return 0
    return 0


def load_causal(mid=None):
    mid = mid or model_id()
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if torch.cuda.is_available():
        vram = _vram_bytes()
        force_4bit = os.environ.get("LLM_4BIT", "0") == "1"
        if force_4bit or (0 < vram < FOURBIT_VRAM):
            print(f"[loader] {mid} GPU {vram / 1024**3:.1f}GB -> 4bit 量化加载", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, load_in_4bit=True, device_map="auto", trust_remote_code=True)
        else:
            print(f"[loader] {mid} GPU {vram / 1024**3:.1f}GB -> float16 加载", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    else:
        print(f"[loader] {mid} CPU float32 加载", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float32, trust_remote_code=True)
    model.eval()
    return tok, model


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
