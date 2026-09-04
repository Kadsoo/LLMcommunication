"""Step10 原型: Delta KV — 增量同步的压缩比与等价性.
场景: A机prefill后生成K步, 只传新增的K个位置的KV(delta), B机本地已有历史KV, 拼接续生成.
验证: 不同prompt长度下 压缩比 = 全量KV/deltaKV, 且B端续生成与A端直推逐token一致.
适配 transformers 5.x DynamicCache.
"""
import sys, time, io, copy
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from telemetry.logger import log

MODEL = "Qwen/Qwen3-0.6B"
K_STEP, M_STEP = 16, 24
LENGTHS = [50, 100, 200, 400, 800]

BASE_TEXT = ("The quick brown fox jumps over the lazy dog. Machine learning systems process "
             "information through neural networks that learn patterns from data. ")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True)
model.eval()


def pad_prompt(n):
    ids = tok(BASE_TEXT, return_tensors="pt")["input_ids"]
    while ids.shape[1] < n:
        ids = torch.cat([ids, ids], dim=1)
    return ids[:, :n]


def to_pure(kv):
    """DynamicCache -> tuple((keys, values), ...) 用于字节统计."""
    return tuple((kv.layers[i].keys, kv.layers[i].values) for i in range(len(kv.layers)))


def kv_bytes(kv):
    """KV张量原始字节数 (numel x itemsize), 网络传输理论值."""
    return sum(t.numel() * t.element_size() for layer in to_pure(kv) for t in layer)


def step_next(input_id, kv):
    with torch.no_grad():
        o = model(input_ids=input_id, past_key_values=kv, use_cache=True)
    nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return nxt, o.past_key_values


def gen_steps(input_id, kv, n):
    """从input_id(单token)续n步; deepcopy隔离, 避免原地更新污染外部kv."""
    kv = copy.deepcopy(kv)
    seq = input_id
    for _ in range(n):
        nxt, kv = step_next(seq[:, -1:], kv)
        seq = torch.cat([seq, nxt], dim=1)
    return seq, kv


def take_delta(kv, keep):
    """deepcopy后只留seq维最后keep个位置(新增增量)."""
    d = copy.deepcopy(kv)
    for i in range(len(d.layers)):
        d.layers[i].keys = d.layers[i].keys[:, :, -keep:, :]
        d.layers[i].values = d.layers[i].values[:, :, -keep:, :]
    return d


def cat_kv(hist, delta):
    """hist(DynamicCache) 末尾拼接 delta(仅新位置)."""
    m = copy.deepcopy(hist)
    for i in range(len(m.layers)):
        m.layers[i].keys = torch.cat([m.layers[i].keys, delta.layers[i].keys], dim=2)
        m.layers[i].values = torch.cat([m.layers[i].values, delta.layers[i].values], dim=2)
    return m


def main():
    rows = []
    for L in LENGTHS:
        prompt = pad_prompt(L)
        # A端: prefill 全量KV
        with torch.no_grad():
            kv0 = model(prompt, use_cache=True).past_key_values
        s_full = kv_bytes(kv0)
        # A端: 续生成K_STEP步
        seq, kvK = gen_steps(prompt[:, -1:], kv0, K_STEP)
        # delta = 新增K_STEP个位置
        delta = take_delta(kvK, K_STEP)
        s_delta = kv_bytes(delta)
        ratio = s_full / max(s_delta, 1)
        # B端: 历史KV + delta 拼接
        kv_merge = cat_kv(kv0, delta)
        # 等价性: B端从A端最后token续M_STEP vs A端直推
        ref, _ = gen_steps(seq[:, -1:], kvK, M_STEP)
        b_out, _ = gen_steps(seq[:, -1:], kv_merge, M_STEP)
        equiv = torch.equal(ref, b_out)
        rows.append({"L": L, "full_b": s_full, "delta_b": s_delta, "ratio": round(ratio, 1), "equiv": equiv})
        log("delta_kv", prompt_len=L, full_bytes=s_full, delta_bytes=s_delta,
            ratio=round(ratio, 1), equiv=equiv)
        print(f"L={L:4d}  full={s_full:9d}B  delta={s_delta:7d}B  压缩比={ratio:6.1f}x  equiv={equiv}", flush=True)

    md = ["## Step10 原型: Delta KV 增量同步", "| Prompt长度 | 全量KV | Delta KV | 压缩比 | 等价性 |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['L']} | {r['full_b']}B | {r['delta_b']}B | {r['ratio']}x | {'OK' if r['equiv'] else 'FAIL'} |")
    md.append("\n注: Delta=仅新增生成位置的KV; 历史KV双方本地已有无需传输。")
    (Path(ROOT) / "eval" / "delta_kv_table.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    assert all(r["equiv"] for r in rows), "Delta拼接续生成与直推不一致"
    print("Step10 Delta KV 等价性全部通过")


if __name__ == "__main__":
    main()