"""Transport 收发小助手: 按类型发送, 接收时拆出 (type, bytes, meta)."""
import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, ROOT)
from transport.adapter import Message, TOKEN, KV

def send_text(tp, text: str, **meta):
    tp.send(Message.make(TOKEN, text.encode("utf-8"), meta))

def send_kv(tp, kv_bytes: bytes, **meta):
    tp.send(Message.make(KV, kv_bytes, meta))

def recv_payload(tp, timeout=None):
    m = tp.recv(timeout=timeout)
    return m["type"], Message.data(m), m.get("meta", {})
