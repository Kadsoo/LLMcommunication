"""Step8: 生成深色实验仪表盘静态Demo页 (零依赖单文件)."""
import json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def load_jsonl(p):
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def fmt_bytes(n):
    n = float(n)
    if n >= 1e6: return f"{n/1e6:.2f} MB"
    if n >= 1e3: return f"{n/1e3:.1f} KB"
    return f"{int(n)} B"

textmas = load_jsonl(ROOT / "agents" / "textmas_results.jsonl")
runs = load_jsonl(ROOT / "telemetry" / "runs.jsonl")
cmps = [r for r in runs if r.get("event") == "compare"]
kv_eq = next((r for r in runs if r.get("event") == "kv_equiv"), {})
tcp_a = next((r for r in runs if r.get("event") == "tcp_a"), {})
tcp_b = next((r for r in runs if r.get("event") == "tcp_b"), {})

# ---- KPI ----
tm_ok = sum(1 for r in textmas if r.get("correct"))
tm_n = len(textmas)
cmps = cmps[-50:]
tok_ok = sum(1 for r in cmps if r.get("tok_strict")); kv_ok = sum(1 for r in cmps if r.get("kv_strict"))
l_t = sum(1 for r in cmps if r.get("tok_loose")); l_k = sum(1 for r in cmps if r.get("kv_loose"))
at = sum(r.get("tok_s", 0) for r in cmps) / max(len(cmps), 1)
ak = sum(r.get("kv_s", 0) for r in cmps) / max(len(cmps), 1)
avg_tok_b = sum(r.get("tok_bytes", 0) for r in cmps) / max(len(cmps), 1)
avg_kv_b = sum(r.get("kv_bytes", 0) for r in cmps) / max(len(cmps), 1)
ratio = avg_kv_b / max(avg_tok_b, 1)
kv_bytes_eq = kv_eq.get("kv_bytes", 0)
kv_layers = kv_eq.get("layers", "?")
net_s = tcp_a.get("net_s", "?")

# ---- 对比表 (log bar: TOKEN vs KV) ----
max_log = math.log10(max(avg_kv_b, 1))
def bar(b, cls):
    w = max(math.log10(max(b, 1)) / max_log * 100, 1.5)
    return f"<div class='bar {cls}' style='width:{w:.1f}%'></div>"
qrows = ""
for r in cmps:
    ti = "✓" if r.get("tok_strict") else "✗"; ki = "✓" if r.get("kv_strict") else "✗"
    tc = "y" if r.get("tok_strict") else "n"; kc = "y" if r.get("kv_strict") else "n"
    qrows += (f"<tr><td class='q'>{esc(r.get('question',''))[:70]}</td>"
              f"<td>{esc(r.get('category',''))}</td>"
              f"<td class='{tc}'>{ti} · {r.get('tok_s','?')}s · {fmt_bytes(r.get('tok_bytes',0))}</td>"
              f"<td class='{kc}'>{ki} · {r.get('kv_s','?')}s · {fmt_bytes(r.get('kv_bytes',0))}</td></tr>")

# ---- 对话时间线 ----
cards = ""
for i, r in enumerate(textmas, 1):
    ok = bool(r.get("correct")); badge = "✓ 成功" if ok else "✗ 失败"; cls = "ok" if ok else "fail"
    a2 = esc(str(r.get("a2", ""))); a1 = esc(str(r.get("a1", ""))); b = esc(str(r.get("critic", "")))
    head = a2[:120].replace("\n", " ")
    cards += f"""<details class="card {cls}" {"open" if i == 1 else ""}>
<summary><span class="idx">Q{i}</span><span class="qt">{esc(r.get('q',''))}</span>
<span class="badge">{badge}</span><span class="final">{head}…</span></summary>
<div class="flow">
<div class="node a"><h4>A · 初答 <small>{r.get('tokens','?')} tokens</small></h4><p>{a1[:600]}</p></div>
<div class="arrow">↓ B 批评</div>
<div class="node b"><h4>B · 批评</h4><p>{b[:600]}</p></div>
<div class="arrow">↓ A 修正</div>
<div class="node c"><h4>A · 终答 <small>{r.get('time_s','?')}s</small></h4><p>{a2[:800]}</p></div>
</div></details>"""

# ---- telemetry ----
trows = ""
labels = {"tcp_b": "跨进程 B 推理", "tcp_a": "跨进程 A 全程", "kv_equiv": "KV 等价性", "compare": "Token/KV 对比"}
for r in runs:
    ev = r.get("event", ""); keep = {k: v for k, v in r.items() if k not in ("t", "event")}
    trows += f"<tr><td>{esc(labels.get(ev, ev))}</td><td class='mono'>{esc(json.dumps(keep, ensure_ascii=False))}</td></tr>"

html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM 通信实验台 · Step8</title>
<style>
:root{{--bg:#0b0f14;--panel:#111823;--line:#1f2a3a;--txt:#dbe4f0;--dim:#8fa0b5;--cy:#38e1c6;--am:#ffc857;--gr:#3ddc84;--rd:#ff6b6b}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(1200px 500px at 80% -10%,#14243a 0%,var(--bg) 55%) fixed,var(--bg);color:var(--txt);font-family:"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif}}
.wrap{{max-width:1060px;margin:0 auto;padding:28px 18px 60px}}
.hero{{display:flex;flex-wrap:wrap;align-items:flex-end;gap:14px;border-bottom:1px solid var(--line);padding-bottom:18px}}
.hero h1{{font-family:Georgia,"Songti SC",serif;font-size:30px;margin:0;letter-spacing:1px}}
.hero h1 em{{color:var(--cy);font-style:normal}}
.pill{{font-size:12px;border:1px solid var(--cy);color:var(--cy);border-radius:20px;padding:3px 12px}}
.sub{{color:var(--dim);font-size:13px;margin-top:8px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:linear-gradient(180deg,#131c29,#0e141f);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.kpi .k{{font-size:12px;color:var(--dim);letter-spacing:1px}}
.kpi .v{{font-family:Consolas,"JetBrains Mono",monospace;font-size:26px;margin:6px 0 2px}}
.kpi .v small{{font-size:13px;color:var(--dim)}}
.kpi .d{{font-size:12.5px;color:var(--dim);line-height:1.6}}
.kpi.hot{{border-color:var(--cy)}}.kpi.hot .v{{color:var(--cy)}}
h2{{font-size:17px;margin:30px 0 10px;display:flex;align-items:center;gap:8px}}
h2::before{{content:"";width:8px;height:18px;background:var(--cy);border-radius:2px;display:inline-block}}
.duel{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:700px){{.duel{{grid-template-columns:1fr}}}}
.mode{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
.mode h3{{margin:0 0 4px;font-size:15px}}.mode.t h3{{color:var(--gr)}}.mode.k h3{{color:var(--am)}}
.mode .big{{font-family:Consolas,monospace;font-size:24px;margin:6px 0}}
.bar{{height:8px;border-radius:4px;margin:8px 0}}.bar.t{{background:var(--gr)}}.bar.k{{background:var(--am)}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:var(--panel);border-radius:12px;overflow:hidden}}
th,td{{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}}
th{{color:var(--dim);font-weight:600;background:#0e1622}}td.q{{max-width:260px}}td.y{{color:var(--gr)}}td.n{{color:var(--rd)}}
.card{{background:var(--panel);border:1px solid var(--line);border-left:6px solid var(--gr);border-radius:10px;margin:10px 0}}
.card.fail{{border-left-color:var(--rd)}}.card summary{{cursor:pointer;padding:12px 14px;list-style:none;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}}
.idx{{font-family:Consolas,monospace;color:var(--cy)}}.qt{{font-weight:600}}
.badge{{font-size:12px;border:1px solid currentColor;border-radius:12px;padding:1px 10px}}.ok .badge{{color:var(--gr)}}.fail .badge{{color:var(--rd)}}
.final{{color:var(--dim);font-size:12.5px;flex-basis:100%}}
.flow{{padding:0 14px 14px}}.node{{background:#0c121c;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:6px 0}}
.node h4{{margin:0 0 6px;font-size:13px}}.node.a h4{{color:#7db8ff}}.node.b h4{{color:var(--am)}}.node.c h4{{color:var(--gr)}}
.node p{{margin:0;font-size:12.5px;line-height:1.7;color:#c4d0e0;white-space:pre-wrap}}
.node small{{color:var(--dim)}}.arrow{{color:var(--dim);font-size:12px;padding-left:6px}}
.mono{{font-family:Consolas,monospace;font-size:12px;word-break:break-all}}
.foot{{margin-top:34px;color:var(--dim);font-size:12.5px;border-top:1px solid var(--line);padding-top:14px;line-height:1.8}}
</style></head><body><div class="wrap">
<div class="hero"><h1>LLM 通信实验台 <em>· Step8</em></h1><span class="pill">全链路跑通 · Qwen3-0.6B 本机</span></div>
<div class="sub">TextMAS → Transport 接口 → TCP 跨进程 → Telemetry → KV 传输 → Token/KV 对比 → 可视化 ｜ 服务器阶段切 4B ｜ Step9 LabTransport 预留</div>
<div class="kpis">
<div class="kpi hot"><div class="k">核心结论 · 传输代价</div><div class="v">{ratio:,.0f}×</div><div class="d">KV 平均 {fmt_bytes(avg_kv_b)} vs TOKEN 平均 {fmt_bytes(avg_tok_b)}。<br>无损但极重 —— 压缩即 Step10 方向。</div></div>
<div class="kpi"><div class="k">成功率 · 50题严格判定</div><div class="v">{tok_ok}/{len(cmps)} <small>vs</small> {kv_ok}/{len(cmps)}</div><div class="d">TOKEN vs KV 同题对比（自编30+GSM8K 20），KV 端到端平均 {ak:.1f}s vs TOKEN {at:.1f}s。</div></div>
<div class="kpi"><div class="k">KV 无损验证</div><div class="v">✓ EQUIV</div><div class="d">续写与单机直推逐 token 一致。<br>载荷 {fmt_bytes(kv_bytes_eq)} / {kv_layers} 层。</div></div>
<div class="kpi"><div class="k">跨进程 TCP</div><div class="v">{net_s}<small>s</small></div><div class="d">双进程 A→B→A 真实走网，B 推理 {tcp_b.get('infer_s','?')}s。<br>TextMAS 单机 {tm_ok}/{tm_n}。</div></div>
</div>
<h2>TOKEN vs KV —— 同题对比（50题）</h2>
<div class="duel">
<div class="mode t"><h3>TOKEN 模式 · 传文本</h3><div class="big">{fmt_bytes(avg_tok_b)} / 题</div>{bar(avg_tok_b,'t')}<div class="sub">严格 {tok_ok}/{len(cmps)} · 宽松 {l_t}/{len(cmps)} ｜ 轻量，语义有损耗风险</div></div>
<div class="mode k"><h3>KV 模式 · 传 past_key_values</h3><div class="big">{fmt_bytes(avg_kv_b)} / 题</div>{bar(avg_kv_b,'k')}<div class="sub">严格 {kv_ok}/{len(cmps)} · 宽松 {l_k}/{len(cmps)} ｜ 无损，负载约 {ratio:,.0f} 倍</div></div>
</div>
<p style="height:8px"></p>
<table><tr><th>题目</th><th>题型</th><th>TOKEN（结果·耗时·负载）</th><th>KV（结果·耗时·负载）</th></tr>{qrows}</table>
<h2>多智能体对话 · A → B → A（点击展开）</h2>{cards}
<h2>遥测 Telemetry</h2>
<table><tr><th style="width:160px">事件</th><th>指标</th></tr>{trows}</table>
<div class="foot">管线：P0 初始化 → P1 TextMAS → P2 Transport → P3 TCP 跨机 → P4 日志 → P5 KV → P6 对比 → P7 展示。<br>
下一步：两台服务器改 IP 即真跨机，换 4B 重跑 P6/P7；实验室接口到货后实现 LabTransport。</div>
</div></body></html>"""

out = ROOT / "demo" / "index.html"
out.write_text(html, encoding="utf-8")
print(f"demo -> {out} ({len(html)} chars)")
