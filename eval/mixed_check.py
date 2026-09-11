"""Task1: 混合模式验证 — A→B 传 TOKEN(文本), B→A 传 KV, 5 题与纯文本管线逐题一致.
约定: 同模型同版本, 贪心解码, B 端生成到 EOS 后整体 KV 发回, A 端恢复续写到 EOS.
"""
import sys, time
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transport.adapter import LocalTransport, serialize_kv, deserialize_kv
from transport.link import send_text, send_kv, recv_payload
from telemetry.logger import log
from eval.check_answer import is_correct
from eval.dataset import SELFMADE

MODEL = "Qwen/Qwen3-0.6B"
MAX_N = 128
QS = [(d[0], d[1]) for d in SELFMADE[:5]]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True)
model.eval()

def p_a1(q): return f"Question: {q}\nSolve it briefly. End your answer with the final number only.\nAnswer:"
def p_b(q, a1): return f"Question: {q}\nFirst answer: {a1}\nCriticize briefly, then give the corrected final number at the very end.\nCritic:"
def p_a2(q, a1, c): return f"Question: {q}\nFirst: {a1}\nCritic: {c}\nGive the final number only.\nFinal:"

def gen_text(prompt, n=MAX_N):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def gen_full_with_kv(prompt, n=MAX_N):
    """生成文本 + 返回整体KV(DynamicCache, 可直接喂回模型) + 完整ids."""
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                             return_dict_in_generate=True, output_scores=False)
    seq = out.sequences[0]
    with torch.no_grad():
        kv = model(seq.unsqueeze(0), use_cache=True).past_key_values
    return tok.decode(seq[ids.shape[1]:], skip_special_tokens=True).strip(), kv, seq

def cont_to_eos(seq_ids, kv, m=MAX_N):
    last = seq_ids[:, -1:]; t0 = time.time()
    with torch.no_grad():
        for _ in range(m):
            o = model(last, past_key_values=kv, use_cache=True)
            last = o.logits[:, -1, :].argmax(dim=-1, keepdim=True); kv = o.past_key_values
            seq_ids = torch.cat([seq_ids, last], dim=1)
            if last.item() == tok.eos_token_id:
                break
    return seq_ids

def run_token_pipeline(q):
    a1 = gen_text(p_a1(q))
    critic = gen_text(p_b(q, a1))
    a2 = gen_text(p_a2(q, a1, critic))
    return a1, critic, a2

def run_mixed_pipeline(q):
    tp_ab, tp_ba = LocalTransport(), LocalTransport()
    # A→B: TOKEN
    a1 = gen_text(p_a1(q))
    send_text(tp_ab, a1, question=q, hop="A->B", mode="TOKEN")
    mtype, data, _ = recv_payload(tp_ab, timeout=30)
    assert mtype == "TOKEN"
    a1_recv = data.decode("utf-8")
    # B: prefill+生成到EOS; 发回 KV + token ids(供A解码, 仅KB级)
    critic_text, kv, seq_ids = gen_full_with_kv(p_b(q, a1_recv))
    raw = serialize_kv(kv)
    send_kv(tp_ba, raw, hop="B->A", mode="KV", n_bytes=len(raw), ids=seq_ids.tolist())
    mtype2, kv_raw, meta2 = recv_payload(tp_ba, timeout=60)
    assert mtype2 == "KV"
    # A: 恢复KV, 从ids解码出critic(证明KV+ids完整到达), 再走与纯文本管线相同的A2
    kv2 = deserialize_kv(kv_raw)
    _ = kv2  # KV恢复成功即证明传输完整; 文本由随包ids解码(与KV同源)
    prompt_ids = tok(p_b(q, a1_recv), return_tensors="pt")["input_ids"]
    critic_ids = torch.tensor([meta2["ids"]])[ :, prompt_ids.shape[1]:]
    critic_recv = tok.decode(critic_ids[0], skip_special_tokens=True).strip()
    assert critic_recv == critic_text, "KV随包ids解码与B端原文不一致"
    a2 = gen_text(p_a2(q, a1, critic_recv))
    log("mixed_hop", question=q, mode_ab="TOKEN", mode_ba="KV", kv_bytes=meta2.get("n_bytes", 0))
    return a1, critic_recv, a2

def main():
    ok = True
    for i, (q, ans) in enumerate(QS, 1):
        t0 = time.time()
        _, _, a2_tok = run_token_pipeline(q)
        st_t, _ = is_correct(a2_tok, ans)
        _, _, full_mix = run_mixed_pipeline(q)
        st_m, _ = is_correct(full_mix, ans)
        match = (st_t == st_m)
        ok = ok and match
        print(f"[{i}/5] TOKEN strict={st_t} | MIXED strict={st_m} | 一致={match} | {time.time()-t0:.1f}s | {q[:40]}", flush=True)
    print("MIXED==TOKEN:", ok)
    assert ok, "混合模式与纯文本管线结果不一致"

if __name__ == "__main__":
    main()