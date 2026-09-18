"""P1 单机 TextMAS: A解答 -> B批评 -> A修正 (Qwen/Qwen3-0.6B, CPU可跑)

0.6B 撑不起长链复述, 这里用短答提示词(与 eval/compare.py 同风格):
每段只给简短作答并以纯数字结尾, 避免车轱辘复读与幻觉发散.
判分用 eval/check_answer.py 的严格(末尾数字)/宽松(全文含数字), 不再用子串包含.
"""
import json, time, sys, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.check_answer import is_correct

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
def gen(tok, model, prompt, max_new=64):
    inputs = tok(prompt, return_tensors="pt")
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         repetition_penalty=1.1)
    dt = time.time() - t0
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip(), dt, len(out[0]) - inputs["input_ids"].shape[1]

def run():
    tok, model = build()
    OUT.write_text("", encoding="utf-8", newline="\n")
    for item in QUESTIONS:
        q = item["q"]
        a1, t1, n1 = gen(tok, model, f"Question: {q}\nSolve it briefly. End your answer with the final number only.\nAnswer:")
        b, t2, n2 = gen(tok, model, f"Question: {q}\nProposed answer: {a1}\nCheck it briefly. Start with Verdict: CORRECT or Verdict: WRONG, then one short sentence.\nCritic:")
        a2, t3, n3 = gen(tok, model, f"Question: {q}\nFirst answer: {a1}\nCritic: {b}\nGive the corrected final answer briefly, ending with the final number only.\nFinal:")
        st, lo = is_correct(a2, item["answer"])
        rec = {"q": q, "expected": item["answer"], "a1": a1, "critic": b, "a2": a2,
               "correct": st, "strict": st, "loose": lo,
               "time_s": round(t1+t2+t3, 2), "tokens": n1+n2+n3}
        print(f"Q: {q}\nA1: {a1[:200]}\nB: {b[:200]}\nA2: {a2[:200]}\nstrict={st} loose={lo}\n---")
        with open(OUT, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    run()
