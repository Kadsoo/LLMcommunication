"""答案判定: 严格(末尾数字) / 宽松(全文含数字), 支持小数与千分位."""
import re

_NUM_RE = re.compile(r"-?\d+(?:[,.]\d+)*")
_STRIP = str.maketrans("", "", ",$元¥￥% km米小时dollarsyuan")


def _norm(s: str) -> str:
    return s.strip().translate(_STRIP).lower()


def extract_numbers(text: str):
    """抽全部数字, 千分位逗号去掉, 统一为float字符串."""
    out = []
    for m in _NUM_RE.findall(text):
        t = m.replace(",", "")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def strict_match(text: str, answer: str) -> bool:
    """严格: 文本最后一个数字 == 答案."""
    nums = extract_numbers(text)
    if not nums:
        return False
    try:
        return nums[-1] == float(_norm(answer))
    except ValueError:
        return _norm(answer) in _norm(text)


def loose_match(text: str, answer: str) -> bool:
    """宽松: 全文任一数字 == 答案."""
    nums = extract_numbers(text)
    try:
        target = float(_norm(answer))
    except ValueError:
        return _norm(answer) in _norm(text)
    return any(n == target for n in nums)


def is_correct(text: str, answer: str):
    return strict_match(text, answer), loose_match(text, answer)


if __name__ == "__main__":
    cases = [
        ("答案是 14", "14", (True, True)),
        ("最终答案: 8 apples", "8", (True, True)),
        ("有6个苹果，但总数是14", "14", (True, True)),
        ("有6个苹果，但总数是14", "6", (False, True)),
        ("The final answer is 180 km.", "180", (True, True)),
        ("$6 left", "6", (True, True)),
        ("2,500 meters", "2500", (True, True)),
        ("0.5 kg is 500 g", "500", (True, True)),
        ("I don't know.", "5", (False, False)),
        ("答案是 1.5", "1.5", (True, True)),
        ("答案约等于14左右", "14", (True, True)),
    ]
    fails = 0
    for text, ans, exp in cases:
        got = is_correct(text, ans)
        mark = "OK " if got == exp else "FAIL"
        if got != exp: fails += 1
        print(f"{mark} strict={got[0]} loose={got[1]} (exp {exp}) <- {text!r} ans={ans}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)