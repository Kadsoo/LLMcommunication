"""Step5 Telemetry: 推理/传输/负载统一记录 JSONL."""
import json, time
from pathlib import Path

LOG = Path(__file__).parent / "runs.jsonl"

def log(event: str, **fields):
    rec = {"t": round(time.time(), 3), "event": event, **fields}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
