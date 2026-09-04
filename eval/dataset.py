"""Step7 数据集: 自编30题(5类) + GSM8K 20题."""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "gsm8k_test.jsonl"

SELFMADE = [
    # 算术 (答案整数)
    ("12 + 7 = ?", "19", "arithmetic", "selfmade"),
    ("8 * 6 = ?", "48", "arithmetic", "selfmade"),
    ("100 - 37 = ?", "63", "arithmetic", "selfmade"),
    ("15 / 3 = ?", "5", "arithmetic", "selfmade"),
    ("7 + 3 * 5 = ?", "22", "arithmetic", "selfmade"),
    ("2^3 + 4 = ?", "12", "arithmetic", "selfmade"),
    # 应用题
    ("Janet has 3 apples and buys 5 more. How many apples does she have in total?", "8", "word_problem", "selfmade"),
    ("A train travels 60 km per hour. How far does it go in 3 hours?", "180", "word_problem", "selfmade"),
    ("Tom has 10 dollars and spends 4. How much does he have left?", "6", "word_problem", "selfmade"),
    ("Debra sees 30 bees leave, and half that many return. How many bees return?", "15", "word_problem", "selfmade"),
    ("A bookstore has 24 books and puts them equally on 6 shelves. How many books per shelf?", "4", "word_problem", "selfmade"),
    ("A bakery makes 36 cookies and packs them into boxes of 9. How many boxes are needed?", "4", "word_problem", "selfmade"),
    # 比例
    ("A car drives 12 km per liter of fuel. How far can it drive on 5 liters?", "60", "ratio", "selfmade"),
    ("One dozen is 12 items. How many items are in 4 dozen?", "48", "ratio", "selfmade"),
    ("A recipe for 4 people needs 2 cups of flour. How many cups for 8 people?", "4", "ratio", "selfmade"),
    ("A 6-meter rope is cut into 0.5-meter pieces. How many pieces are there?", "12", "ratio", "selfmade"),
    ("Someone types 60 words per minute. How many words in 5 minutes?", "300", "ratio", "selfmade"),
    ("Each apple costs 3 yuan. How much do 5 apples cost?", "15", "ratio", "selfmade"),
    # 逻辑
    ("There are 3 boxes and each box contains 2 balls. How many balls in total?", "6", "logic", "selfmade"),
    ("Ming is 3rd in line, and XiaoHong is 2 places behind him. What position is XiaoHong?", "5", "logic", "selfmade"),
    ("5 people each shake hands with each other once. How many handshakes in total?", "10", "logic", "selfmade"),
    ("An elevator starts at floor 8 and goes down 3 floors. Which floor is it at?", "5", "logic", "selfmade"),
    ("It is 9 o'clock now. What time is it 3 hours later?", "12", "logic", "selfmade"),
    ("How many days are in 3 weeks?", "21", "logic", "selfmade"),
    # 单位换算
    ("1 meter is 100 cm. How many cm is 3 meters?", "300", "units", "selfmade"),
    ("1 kg is 1000 g. How many grams is 0.5 kg?", "500", "units", "selfmade"),
    ("1 hour is 60 minutes. How many hours is 120 minutes?", "2", "units", "selfmade"),
    ("1 km is 1000 m. How many meters is 2.5 km?", "2500", "units", "selfmade"),
    ("1 day is 24 hours. How many hours is 3 days?", "72", "units", "selfmade"),
    ("1 liter is 1000 ml. How many ml is 1.5 liters?", "1500", "units", "selfmade"),
]

GSM8K_URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"
GSM8K_N = 20
SEED = 42


def _fetch_gsm8k():
    import urllib.request
    import io
    import pyarrow.parquet as pq
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        print("downloading GSM8K test set ...", flush=True)
        with urllib.request.urlopen(GSM8K_URL, timeout=120) as r:
            data = r.read()
        tbl = pq.read_table(io.BytesIO(data))
        rows = [{"question": q, "answer": a} for q, a in
                zip(tbl["question"].to_pylist(), tbl["answer"].to_pylist())]
        with open(CACHE, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = []
    rng = random.Random(SEED)
    for r in rng.sample(rows, GSM8K_N):
        q = r["question"].strip().replace("\n", " ")
        ans = r["answer"].strip().split("####")[-1].strip()
        picked.append((q, ans, "gsm8k", "gsm8k"))
    return picked


def load_dataset():
    items = [dict(question=q, answer=a, category=c, source=s) for q, a, c, s in SELFMADE]
    items += [dict(question=q, answer=a, category="gsm8k", source=s) for q, a, c, s in _fetch_gsm8k()]
    return items


if __name__ == "__main__":
    ds = load_dataset()
    print(f"total={len(ds)}")
    from collections import Counter
    print(Counter(d["category"] for d in ds))