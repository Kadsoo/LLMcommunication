"""Step8 交互版 (运行: python demo/app.py -> http://127.0.0.1:7860)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

CSS = """
body, .gradio-container { background: #0b0f14 !important; color: #dbe4f0 !important; }
h1, h2, h3 { color: #dbe4f0 !important; }
.kpi { display: flex; gap: 12px; flex-wrap: wrap; }
.kpi div { background: #111823; border: 1px solid #1f2a3a; border-radius: 10px; padding: 10px 16px; min-width: 180px; }
.kpi b { color: #38e1c6; font-size: 20px; font-family: Consolas, monospace; }
"""

def load():
    def jl(p):
        p = ROOT / p
        if not p.exists(): return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    textmas = jl("agents/textmas_results.jsonl")
    runs = jl("telemetry/runs.jsonl")
    cmps = [r for r in runs if r.get("event") == "compare"]
    kv = next((r for r in runs if r.get("event") == "kv_equiv"), {})
    def fmt(n):
        n = float(n or 0)
        return f"{n/1e6:.2f} MB" if n >= 1e6 else (f"{n/1e3:.1f} KB" if n >= 1e3 else f"{int(n)} B")
    tok_ok = sum(1 for r in cmps if r.get("tok_ok")); kv_ok = sum(1 for r in cmps if r.get("kv_ok"))
    ab = sum(r.get("tok_bytes", 0) for r in cmps) / max(len(cmps), 1)
    ak = sum(r.get("kv_bytes", 0) for r in cmps) / max(len(cmps), 1)
    ratio = ak / max(ab, 1)
    tm_ok = sum(1 for r in textmas if r.get("correct"))
    kpi = (f"<div class='kpi'><div>传输代价<br><b>{ratio:,.0f}×</b><br>KV {fmt(ak)} vs TOKEN {fmt(ab)}</div>"
           f"<div>成功率<br><b>{tok_ok}/{len(cmps)} vs {kv_ok}/{len(cmps)}</b><br>TOKEN vs KV 同题</div>"
           f"<div>KV 无损<br><b>{'✓ EQUIV' if kv.get('equiv') else '?'}</b><br>{fmt(kv.get('kv_bytes', 0))} / {kv.get('layers', '?')}层</div>"
           f"<div>TextMAS<br><b>{tm_ok}/{len(textmas)}</b><br>A→B→A 单机</div></div>")
    qrows = [[r.get("question", ""), "✓" if r.get("tok_ok") else "✗", r.get("tok_s", ""),
              fmt(r.get("tok_bytes", 0)), "✓" if r.get("kv_ok") else "✗", r.get("kv_s", ""),
              fmt(r.get("kv_bytes", 0))] for r in cmps]
    drows = [[r.get("q", ""), str(r.get("a1", ""))[:200], str(r.get("critic", ""))[:200],
              str(r.get("a2", ""))[:200], "成功" if r.get("correct") else "失败"] for r in textmas]
    return kpi, qrows, drows

def launch():
    import gradio as gr
    kpi, qrows, drows = load()
    with gr.Blocks(title="LLM通信 Step8") as demo:
        gr.Markdown("# LLM 通信实验台 · Step8\nQwen3-0.6B 本机 ｜ 服务器阶段切 4B ｜ Step9 LabTransport 预留")
        gr.HTML(kpi)
        gr.Markdown("## TOKEN vs KV 同题对比")
        gr.Dataframe(value=qrows, headers=["题目", "TOKEN", "耗时s", "负载", "KV", "耗时s", "负载"], wrap=True)
        gr.Markdown("## 多智能体对话 A→B→A")
        gr.Dataframe(value=drows, headers=["问题", "A初答", "B批评", "A终答", "结果"], wrap=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)

if __name__ == "__main__":
    launch()
