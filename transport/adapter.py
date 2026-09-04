"""Step3 Transport Adapter: Agent与通信解耦, 支持 TOKEN / KV."""
import base64, io, json, queue, socket, struct
import torch

TOKEN, KV = "TOKEN", "KV"

class Message(dict):
    @staticmethod
    def make(mtype, data_b: bytes, meta=None):
        return {"type": mtype, "data_b64": base64.b64encode(data_b).decode(),
                "meta": meta or {}}
    @staticmethod
    def data(m): return base64.b64decode(m["data_b64"])

class Transport:
    def send(self, msg): raise NotImplementedError
    def recv(self, timeout=None): raise NotImplementedError
    def close(self): pass

class LocalTransport(Transport):
    """单进程回环, 用于单机测试."""
    def __init__(self):
        self.q = queue.Queue()
    def send(self, msg): self.q.put(msg)
    def recv(self, timeout=None): return self.q.get(timeout=timeout)

class TcpTransport(Transport):
    """TCP长度前缀帧: 4字节len + JSON. 一端listen一端connect."""
    def __init__(self, host="127.0.0.1", port=5101, is_server=False, accept_timeout=300):
        self.sock = None
        if is_server:
            s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port)); s.listen(1); s.settimeout(accept_timeout)
            self.sock, _ = s.accept(); s.close()
        else:
            self.sock = socket.create_connection((host, port), timeout=60)
    def send(self, msg):
        b = json.dumps(msg).encode()
        self.sock.sendall(struct.pack("!I", len(b)) + b)
    def recv(self, timeout=None):
        if timeout: self.sock.settimeout(timeout)
        hdr = self._recvn(4); (n,) = struct.unpack("!I", hdr)
        return json.loads(self._recvn(n).decode())
    def _recvn(self, n):
        buf = b""
        while len(buf) < n:
            c = self.sock.recv(n - len(buf))
            if not c: raise ConnectionError("tcp closed")
            buf += c
        return buf
    def close(self):
        try: self.sock.close()
        except Exception: pass

class LabTransport(Transport):
    """Step9预留: 实验室通信接口到货后在此实现, 现在抛错防止误用."""
    def send(self, msg): raise NotImplementedError("Step9 LabTransport未接入")
    def recv(self, timeout=None): raise NotImplementedError("Step9 LabTransport未接入")

# ---- KV 序列化 ----
def serialize_kv(past_key_values) -> bytes:
    buf = io.BytesIO(); torch.save(past_key_values, buf); return buf.getvalue()

def deserialize_kv(b: bytes):
    return torch.load(io.BytesIO(b), weights_only=False)
