"""工作台聊天后端: 直接聊(solver自由对话) + 看讨论(A→B→A固定一轮分步展示).
模型经 model_loader 加载, 服务器上 LLM_MODEL 环境变量切换零改动.
线程安全: 全局锁一次一聊. 懒加载: 首次调用时加载模型.
"""
import sys, time, threading
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
import torch
from telemetry.logger import log
from eval.model_loader import load_causal, model_id, model_device

CHAT_MAX_N = 64
DISCUSS_MAX_N = 128
HIST_TURNS = 3

SOLVER_SYS = "You are a helpful math assistant. Answer briefly."
CRITIC_TMPL = ("Question: {q}\nProposed answer: {a1}\nCriticize errors in one short paragraph. "
               "If correct, say CORRECT.\nCritic:")
REVISE_TMPL = ("Question: {q}\nFirst answer: {a1}\nCritic: {b}\n"
               "Give corrected final answer with number.\nFinal:")

_lock = threading.Lock()
_tok = None
_model = None
_dev = None
_loading = False


def status():
    if _model is not None:
        return f"就绪 ({model_id()})"
    if _loading:
        return "模型加载中…"
    return "未加载（首次聊天时自动加载，约1-2分钟）"


def ensure_model():
    global _tok, _model, _dev, _loading
    if _model is not None:
        return
    _loading = True
    try:
        _tok, _model = load_causal()
        _dev = model_device(_model)
    finally:
        _loading = False


def _gen(prompt, n):
    ids = _tok(prompt, return_tensors="pt")["input_ids"].to(_dev); t0 = time.time()
    with torch.no_grad():
        out = _model.generate(input_ids=ids, max_new_tokens=n, do_sample=False)
    txt = _tok.decode(out[0].cpu()[ids.shape[1]:], skip_special_tokens=True).strip()
    return txt, round(time.time() - t0, 2)


def chat(history, user_msg):
    """直接聊: 最近HIST_TURNS轮历史 + solver人设. 返回 (reply, dt).
    history: [(user, bot), ...]"""
    ensure_model()
    with _lock:
        ctx = "\n".join(
            f"User: {u}\nAssistant: {b}" for u, b in history[-HIST_TURNS:])
        prompt = f"{SOLVER_SYS}\n{ctx}\nUser: {user_msg}\nAssistant:" if ctx else f"{SOLVER_SYS}\nUser: {user_msg}\nAssistant:"
        reply, dt = _gen(prompt, CHAT_MAX_N)
        log("chat-live", mode="chat", prompt=user_msg[:200], dt=dt)
        return reply, dt


def discuss(question):
    """看讨论: A初答→B批评→A修正, 返回 [(角色, 文本, 耗时)] 分步展示."""
    ensure_model()
    steps = []
    with _lock:
        a1, t1 = _gen(f"Question: {question}\nSolve it briefly. End your answer with the final number only.\nAnswer:", DISCUSS_MAX_N)
        steps.append(("A · 初答", a1, t1))
        b, t2 = _gen(CRITIC_TMPL.format(q=question, a1=a1), DISCUSS_MAX_N)
        steps.append(("B · 批评", b, t2))
        a2, t3 = _gen(REVISE_TMPL.format(q=question, a1=a1, b=b), DISCUSS_MAX_N)
        steps.append(("A · 终答", a2, t3))
        log("chat-live", mode="discuss", prompt=question[:200],
            dt=round(t1 + t2 + t3, 2))
    return steps
