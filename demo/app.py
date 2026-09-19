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
/* Gradio 6.x Dataframe 的 wrap 参数实测不生效，这里用作用域 CSS 强制换行。
   注意：实测单元格文字直接放在 .cell-wrap 里（没有 span 子元素），所以直接打在 .cell-wrap 上 */
.wrap-table .cell-wrap { white-space: normal !important; text-overflow: clip !important; overflow: visible !important; overflow-wrap: break-word !important; word-break: break-word !important; align-items: flex-start !important; }
.wrap-table .cell-wrap span { white-space: normal !important; text-overflow: clip !important; overflow: visible !important; overflow-wrap: break-word !important; word-break: break-word !important; }
.wrap-table table { table-layout: auto !important; }
.wrap-table td, .wrap-table th { height: auto !important; }
"""

def load():
    def jl(p):
        p = ROOT / p
        if not p.exists(): return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    textmas = jl("agents/textmas_results.jsonl")
    runs = jl("telemetry/runs.jsonl")
    cmps = [r for r in runs if r.get("event") == "compare" and r.get("category")][-50:]
    kv = next((r for r in runs if r.get("event") == "kv_equiv"), {})
    def fmt(n):
        n = float(n or 0)
        return f"{n/1e6:.2f} MB" if n >= 1e6 else (f"{n/1e3:.1f} KB" if n >= 1e3 else f"{int(n)} B")
    tok_ok = sum(1 for r in cmps if r.get("tok_strict")); kv_ok = sum(1 for r in cmps if r.get("kv_strict"))
    l_t = sum(1 for r in cmps if r.get("tok_loose")); l_k = sum(1 for r in cmps if r.get("kv_loose"))
    ab = sum(r.get("tok_bytes", 0) for r in cmps) / max(len(cmps), 1)
    ak = sum(r.get("kv_bytes", 0) for r in cmps) / max(len(cmps), 1)
    ratio = ak / max(ab, 1)
    tm_ok = sum(1 for r in textmas if r.get("correct"))
    kpi = (f"<div class='kpi'><div>核心 · 传输代价<br><b>{ratio:,.0f}×</b><br>KV {fmt(ak)} vs TOKEN {fmt(ab)}</div>"
           f"<div>成功率 · 严格<br><b>{tok_ok}/{len(cmps)} vs {kv_ok}/{len(cmps)}</b><br>TOKEN vs KV 同题 逐题一致</div>"
           f"<div>宽松通过<br><b>{l_t}/{len(cmps)} vs {l_k}/{len(cmps)}</b><br>两模式同数</div>"
           f"<div>KV 无损<br><b>{'✓ EQUIV' if kv.get('equiv') else '?'}</b><br>{fmt(kv.get('kv_bytes', 0))} / {kv.get('layers', '?')}层</div></div>")
    cat_name = {"arithmetic": "算术", "word_problem": "应用题", "ratio": "比例", "logic": "逻辑", "units": "单位换算", "gsm8k": "GSM8K"}
    qrows = [[r.get("question", ""), cat_name.get(r.get("category", ""), r.get("category", "")),
              "✓" if r.get("tok_strict") else "✗", r.get("tok_s", ""),
              fmt(r.get("tok_bytes", 0)), "✓" if r.get("kv_strict") else "✗", r.get("kv_s", ""),
              fmt(r.get("kv_bytes", 0))] for r in cmps]
    drows = [[r.get("q", ""), str(r.get("a1", ""))[:200], str(r.get("critic", ""))[:200],
              str(r.get("a2", ""))[:200], "成功" if r.get("correct") else "失败"] for r in textmas]
    def md(name):
        p = ROOT / "eval" / name
        return p.read_text(encoding="utf-8") if p.exists() else "暂无数据，先跑对应脚本"
    return kpi, qrows, drows, md("dialogue_table.md"), md("role_table.md")

def launch():
    import gradio as gr
    from demo import chat as chatbe
    kpi, qrows, drows, dialogue_md, role_md = load()

    def on_send(msg, history):
        history = history + [[msg, "⏳ 生成中…"]]
        try:
            reply, dt = chatbe.chat([(h[0], h[1]) for h in history[:-1] if h[1] and not h[1].startswith("⏳")], msg)
            history[-1][1] = f"{reply}\n\n（{dt}s）"
        except Exception as e:
            history[-1][1] = f"出错：{e}"
        return history, ""

    def on_discuss(msg, history):
        history = history + [[msg, "⏳ 智能体讨论中（约1分钟）…"]]
        try:
            steps = chatbe.discuss(msg)
            history[-1][1] = "\n\n".join(f"**{role}**（{dt}s）\n{text}" for role, text, dt in steps)
        except Exception as e:
            history[-1][1] = f"出错：{e}"
        return history, ""

    with gr.Blocks(title="LLM通信 Step8") as demo:
        gr.Markdown("# LLM 通信实验台 · Step8\nQwen3-0.6B 本机 · 50题 ｜ 服务器阶段切 4B ｜ Step9 LabTransport 预留")
        gr.HTML(kpi)
        gr.Markdown("## 和智能体聊天")
        gr.Markdown(f"模型状态：{chatbe.status()}（首次聊天自动加载，约1-2分钟）｜直接聊=和解答智能体自由对话｜看讨论=对你刚发的问题跑 A初答→B批评→A修正")
        chatbot = gr.Chatbot(value=[], label="聊天", height=420)
        msg = gr.Textbox(label="输入", placeholder="如：Janet has 3 apples and buys 5 more. How many in total?")
        with gr.Row():
            btn_send = gr.Button("发送", variant="primary")
            btn_discuss = gr.Button("让智能体讨论这个问题")
        btn_send.click(on_send, [msg, chatbot], [chatbot, msg])
        btn_discuss.click(on_discuss, [msg, chatbot], [chatbot, msg])
        gr.Markdown("## 智能体选型与对话模式")
        gr.Markdown(dialogue_md)
        gr.Markdown(role_md)
        gr.Markdown("## TOKEN vs KV 同题对比（最近50题）")
        gr.Dataframe(value=qrows, headers=["题目", "题型", "TOKEN", "耗时s", "负载", "KV", "耗时s", "负载"], wrap=True, elem_classes=["wrap-table"])
        gr.Markdown("## 多智能体对话 A→B→A")
        gr.Dataframe(value=drows, headers=["问题", "A初答", "B批评", "A终答", "结果"], wrap=True, elem_classes=["wrap-table"])
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)

if __name__ == "__main__":
    launch()
