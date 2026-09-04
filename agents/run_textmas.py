"""P1 单机 TextMAS: A解答 -> B批评 -> A修正 (Qwen/Qwen3-0.6B, CPU可跑)"""
import json, time, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"
OUT = Path(__file__).parent / "textmas_results.jsonl"

QUESTIONS = [
    {"q": "Janet has 3 apples, she buys 5 more. How many apples in total?", "answer": "8"},
    {"q": "A train goes 60 km in 1 hour. How far in 3 hours?", "answer": "180"},
    {"q": "Debra sees 30 bees leave, half that many return. How many returned?", "answer": "15"},
    {"q": "Tom has 10 dollars, spends 4. How much left?", "answer": "6"},
    {"q": "2 + 3 * 4 = ?", "answer": "14"},
]

def build(model_id=MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, trust_remote_code=True)
    model.eval()
    return tok, model

@torch.no_grad()
def gen(tok, model, prompt, max_new=128):
    inputs = tok(prompt, return_tensors="pt")
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    dt = time.time() - t0
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip(), dt, len(out[0]) - inputs["input_ids"].shape[1]

def run():
    tok, model = build()
    OUT.write_text("", encoding="utf-8")
    for item in QUESTIONS:
        q = item["q"]
        a1, t1, n1 = gen(tok, model, f"Question: {q}\nSolve step by step and give final number.\nAnswer:")
        b, t2, n2 = gen(tok, model, f"Question: {q}\nProposed answer: {a1}\nCriticize errors in one short paragraph. If correct, say CORRECT.\nCritic:")
        a2, t3, n3 = gen(tok, model, f"Question: {q}\nFirst answer: {a1}\nCritic: {b}\nGive corrected final answer with number.\nFinal:")
        ok = item["answer"] in a2
        rec = {"q": q, "expected": item["answer"], "a1": a1, "critic": b, "a2": a2,
               "correct": ok, "time_s": round(t1+t2+t3, 2), "tokens": n1+n2+n3}
        print(f"Q: {q}\nA1: {a1[:200]}\nB: {b[:200]}\nA2: {a2[:200]}\nOK={ok}\n---")
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    run()
