"""Step8 交互版 + 实时讨论控制台 (运行: python demo/app.py -> http://127.0.0.1:7860).
控制面板: 单机一键AB / 本机A连对端 / 本机B等待; 时间线每2s轮询本地telemetry实时弹出.
"""
import sys, json, time, socket, subprocess, html as htmllib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

CSS = """
body, .gradio-container { background: #0b0f14 !important; color: #dbe4f0 !important; }
h1, h2, h3 { color: #dbe4f0 !important; }
.kpi { display: flex; gap: 12px; flex-wrap: wrap; }
.kpi div { background: #111823; border: 1px solid #1f2a3a; border-radius: 10px; padding: 10px 16px; min-width: 180px; }
.kpi b { color: #38e1c6; font-size: 20px; font-family: Consolas, monospace; }
.tl-card { background: #111823; border: 1px solid #1f2a3a; border-radius: 10px; padding: 10px 14px; margin: 8px 0; }
.tl-card.ok { border-left: 5px solid #3ddc84; } .tl-card.fail { border-left: 5px solid #ff6b6b; }
.tl-card.run { border-left: 5px solid #38e1c6; }
.node { background: #0c121c; border: 1px solid #1f2a3a; border-radius: 8px; padding: 8px 10px; margin: 6px 0; font-size: 13px; }
.b-tok { color: #3ddc84; border: 1px solid #3ddc84; border-radius: 10px; padding: 0 8px; font-size: 11px; }
.b-kv { color: #ffc857; border: 1px solid #ffc857; border-radius: 10px; padding: 0 8px; font-size: 11px; }
.meta { color: #8fa0b5; font-size: 12px; }
"""

CTL = {"procs": [], "running": False, "error": "", "started_at": 0}

def _wait_port(host, port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = socket.create_connection((host, port), timeout=3); s.close(); return True
        except OSError:
            time.sleep(3)
    return False

def _spawn(args):
    logf = open(ROOT / "telemetry" / "discuss_proc.log", "a", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "transport/discuss.py"] + args,
                         cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    CTL["procs"].append(p)
    return p

def start_discussion(role, peer_ip, port, task, n):
    stop_all()
    CTL["error"] = ""; CTL["running"] = True; CTL["started_at"] = time.time()
    port = int(port); n = int(n)
    try:
        if role == "single":
            _spawn(["--role", "b", "--port", str(port)])
            if not _wait_port("127.0.0.1", port, timeout=180):
                raise RuntimeError("B端未在180s内就绪(可能模型加载慢), 请重试")
            _spawn(["--role", "a", "--host", "127.0.0.1", "--port", str(port),
                    "--task", task, "--n", str(n)])
            return "已启动: 单机一键AB, 讨论进行中, 时间线将实时弹出"
        if role == "a":
            _spawn(["--role", "a", "--host", peer_ip, "--port", str(port),
                    "--task", task, "--n", str(n)])
            return f"已启动: 本机A → 连接 {peer_ip}:{port} (30次重连约5分钟, 请确认对端B已启动)"
        _spawn(["--role", "b", "--port", str(port)])
        return f"已启动: 本机B监听 {port} 端口, 等待对端A连接 (需先加载模型约1分钟)"
    except Exception as e:
        CTL["running"] = False
        CTL["error"] = str(e)
        return f"启动失败: {e}"

def stop_all():
    for p in CTL["procs"]:
        try:
            if p.poll() is None: p.terminate()
        except Exception:
            pass
    CTL["procs"] = []; CTL["running"] = False
    return "已停止"

def _jl(name):
    p = ROOT / name
    if not p.exists(): return []
    try:
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return []

def _esc(s):
    return htmllib.escape(str(s))[:800]

def live_timeline():
    """从本地telemetry构建实时时间线(A端discuss_a + B端discuss_b_msg/discuss_b)."""
    runs = _jl("telemetry/runs.jsonl")
    groups = {}
    for r in runs:
        ev = r.get("event")
        if ev == "discuss_a":
            key = (r.get("task"), f"A{r.get('qid')}")
            g = groups.setdefault(key, {"task": r.get("task"), "rounds": {}, "ok": None})
            rd = g["rounds"].setdefault(r.get("round"), {})
            rd.update({"proposal": r.get("proposal", ""), "critic": r.get("critic", ""),
                       "final": r.get("final", ""), "ok": r.get("ok"),
                       "net": r.get("net_s"), "kv": r.get("kv_bytes"), "infer": r.get("infer_s")})
            g["ok"] = r.get("ok")
        elif ev in ("discuss_b_msg", "discuss_b"):
            key = (r.get("task"), f"B{r.get('qid')}")
            g = groups.setdefault(key, {"task": r.get("task"), "rounds": {}, "ok": None})
            rd = g["rounds"].setdefault(r.get("round"), {})
            if ev == "discuss_b_msg":
                rd["question"] = r.get("question", "")
                rd["proposal"] = rd.get("proposal") or r.get("proposal", "")
            else:
                rd["critic"] = rd.get("critic") or r.get("critic", "")
                rd["b_infer"] = r.get("infer_s"); rd["kv"] = r.get("kv_bytes")
                rd["chain"] = r.get("chain", "")
    if not groups:
        return "<p class='meta'>暂无实时讨论 — 在上方控制面板点“开始” (单机选“一键AB”, 跨机选角色+填对方IP)。</p>"
    out = []
    for (task, who), g in sorted(groups.items()):
        tname = "数学" if task == "math" else "迷宫"
        badge = "✓" if g["ok"] else ("✗" if g["ok"] is False else "…")
        cls = "ok" if g["ok"] else ("fail" if g["ok"] is False else "run")
        rounds = []
        for rnd in sorted(g["rounds"]):
            d = g["rounds"][rnd]
            rounds.append(
                f"<div class='node'><b>第{rnd}轮</b> "
                f"A初答 <span class='b-tok'>TOKEN</span><br>{_esc(d.get('proposal',''))}</div>"
                f"<div class='node'>B批评 <span class='b-kv'>KV</span> "
                f"<span class='meta'>{d.get('chain','')} 推理{d.get('b_infer', d.get('infer','?'))}s 负载{fmt_bytes(d.get('kv',0))}</span>"
                f"<br>{_esc(d.get('critic',''))}</div>"
                + (f"<div class='node'>A终答 <span class='meta'>网络{d.get('net','?')}s</span><br>{_esc(d.get('final',''))}</div>" if d.get("final") else ""))
        out.append(f"<div class='tl-card {cls}'><b>{tname} {who}</b> [{badge}]" + "".join(rounds) + "</div>")
    return "".join(out)

def fmt_bytes(n):
    n = float(n or 0)
    return f"{n/1e6:.2f} MB" if n >= 1e6 else (f"{n/1e3:.1f} KB" if n >= 1e3 else f"{int(n)} B")

def status():
    alive = [p for p in CTL["procs"] if p.poll() is None]
    if CTL["error"]:
        return f"❌ {CTL['error']}"
    if alive:
        return f"🟢 运行中 ({len(alive)}个进程, 已运行{int(time.time()-CTL['started_at'])}s)"
    if CTL["procs"]:
        codes = [p.poll() for p in CTL["procs"]]
        tail = ""
        try:
            tail = (ROOT / "telemetry" / "discuss_proc.log").read_text(encoding="utf-8").splitlines()[-3:]
            tail = " | ".join(tail)
        except Exception:
            pass
        return f"⚪ 已结束 (退出码{codes}) {tail}"
    return "⚪ 空闲"

def load():
    def jl(p):
        p = ROOT / p
        if not p.exists(): return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    textmas = jl("agents/textmas_results.jsonl")
    runs = jl("telemetry/runs.jsonl")
    cmps = [r for r in runs if r.get("event") == "compare" and r.get("category")][-50:]
    kv = next((r for r in runs if r.get("event") == "kv_equiv"), {})
    tok_ok = sum(1 for r in cmps if r.get("tok_strict")); kv_ok = sum(1 for r in cmps if r.get("kv_strict"))
    l_t = sum(1 for r in cmps if r.get("tok_loose")); l_k = sum(1 for r in cmps if r.get("kv_loose"))
    ab = sum(r.get("tok_bytes", 0) for r in cmps) / max(len(cmps), 1)
    ak = sum(r.get("kv_bytes", 0) for r in cmps) / max(len(cmps), 1)
    ratio = ak / max(ab, 1)
    tm_ok = sum(1 for r in textmas if r.get("correct"))
    latest_sum = {}
    for s in runs:
        if s.get("event") == "discuss_summary":
            latest_sum[s.get("task")] = s
    disc_s = list(latest_sum.values())
    disc_a = [r for r in runs if r.get("event") == "discuss_a"][-20:]
    ds_txt = " / ".join(f"{'数学' if s.get('task')=='math' else '迷宫'} {s.get('wins')}/{s.get('total')}" for s in disc_s) or "暂无"
    kpi = (f"<div class='kpi'><div>核心 · 传输代价<br><b>{ratio:,.0f}×</b><br>KV {fmt_bytes(ak)} vs TOKEN {fmt_bytes(ab)}</div>"
           f"<div>成功率 · 严格<br><b>{tok_ok}/{len(cmps)} vs {kv_ok}/{len(cmps)}</b><br>TOKEN vs KV 同题 逐题一致</div>"
           f"<div>宽松通过<br><b>{l_t}/{len(cmps)} vs {l_k}/{len(cmps)}</b><br>两模式同数</div>"
           f"<div>跨机讨论<br><b>{ds_txt}</b><br>A→B TOKEN / B→A KV</div>"
           f"<div>KV 无损<br><b>{'✓ EQUIV' if kv.get('equiv') else '?'}</b><br>{fmt_bytes(kv.get('kv_bytes', 0))} / {kv.get('layers', '?')}层</div></div>")
    cat_name = {"arithmetic": "算术", "word_problem": "应用题", "ratio": "比例", "logic": "逻辑", "units": "单位换算", "gsm8k": "GSM8K"}
    qrows = [[r.get("question", "")[:90], cat_name.get(r.get("category", ""), r.get("category", "")),
              "✓" if r.get("tok_strict") else "✗", r.get("tok_s", ""),
              fmt_bytes(r.get("tok_bytes", 0)), "✓" if r.get("kv_strict") else "✗", r.get("kv_s", ""),
              fmt_bytes(r.get("kv_bytes", 0))] for r in cmps]
    drows = [[r.get("q", ""), str(r.get("a1", ""))[:200], str(r.get("critic", ""))[:200],
              str(r.get("a2", ""))[:200], "成功" if r.get("correct") else "失败"] for r in textmas]
    grows = [[("数学" if r.get("task") == "math" else "迷宫") + " Q" + str(r.get("qid", "")),
              "第" + str(r.get("round", "")) + "轮", "✓" if r.get("ok") else "✗",
              "TOKEN→KV", str(r.get("net_s", "")) + "s", fmt_bytes(r.get("kv_bytes", 0))] for r in disc_a]
    return kpi, qrows, drows, grows

def launch():
    import gradio as gr
    kpi, qrows, drows, grows = load()
    with gr.Blocks(title="LLM通信实验台") as demo:
        gr.Markdown("# LLM 通信实验台 · 实时讨论控制台\nQwen3-0.6B ｜ 服务器阶段切 4B ｜ Step9 LabTransport 预留")
        gr.Markdown("## 实时讨论（选角色 → 开始，结果在下方逐轮弹出）")
        with gr.Row():
            role = gr.Radio(["single", "a", "b"], value="single",
                            label="角色", info="single=单机一键AB｜a=本机A连对端｜b=本机B等待连接")
            peer_ip = gr.Textbox("127.0.0.1", label="对方IP（单机不用改，跨机A填B机IP）")
            port = gr.Number(5102, label="端口", precision=0)
            task = gr.Radio(["math", "maze"], value="math", label="任务")
            nq = gr.Number(5, label="题数", precision=0)
        with gr.Row():
            btn_start = gr.Button("开始", variant="primary")
            btn_stop = gr.Button("停止")
        status_box = gr.Textbox("⚪ 空闲", label="状态", interactive=False)
        timeline = gr.HTML(live_timeline())
        btn_start.click(start_discussion, [role, peer_ip, port, task, nq], [status_box])
        btn_stop.click(stop_all, None, [status_box])
        try:
            timer = gr.Timer(2.0)
            timer.tick(lambda: (live_timeline(), status()), outputs=[timeline, status_box])
        except Exception:
            gr.Button("刷新时间线").click(lambda: (live_timeline(), status()), outputs=[timeline, status_box])
        gr.Markdown("对端要求: 同一局域网, 都已 git clone + 装好环境 + 下好同一模型, transformers 同版本, 防火墙放行讨论端口。")
        gr.HTML(kpi)
        gr.Markdown("## TOKEN vs KV 同题对比（最近50题）")
        gr.Dataframe(value=qrows, headers=["题目", "题型", "TOKEN", "耗时s", "负载", "KV", "耗时s", "负载"], wrap=True)
        gr.Markdown("## 跨机讨论历史（混合模式：A→B TOKEN / B→A KV）")
        gr.Dataframe(value=grows, headers=["题目", "轮次", "结果", "通信模式", "网络耗时", "KV负载"], wrap=True)
        gr.Markdown("## 多智能体对话 A→B→A")
        gr.Dataframe(value=drows, headers=["问题", "A初答", "B批评", "A终答", "结果"], wrap=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)

if __name__ == "__main__":
    launch()
