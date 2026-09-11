# LLMcommunication — 大模型智能通信课程实验

多智能体协作 + 跨机通信实验：两个 LLM Agent 通过文本（TOKEN）或 KV Cache
直接传递中间状态完成协作，对比两种通信方式的正确率、时延与传输代价。

参考论文：LatentMAS（`Latent Collaboration in Multi-Agent Systems.pdf`，纯隐空间协作，
指导 Step6 / Step10）。

## 快速开始

```bash
git clone https://github.com/Kadsoo/LLMcommunication.git
cd LLMcommunication
pip install torch transformers accelerate safetensors pyarrow gradio

python agents/run_textmas.py   # P1: 单机 TextMAS（A解答→B批评→A修正）
python eval/kv_equiv.py        # P5: KV 捕获→传输→恢复，等价性验证
python eval/compare.py         # P6: 50 题 TOKEN vs KV 对比（CPU 约 30 分钟）
python demo/build_report.py    # 生成静态报告 demo/index.html
python demo/app.py             # 交互工作台 http://127.0.0.1:7860
```

## 目录结构

```
agents/        run_textmas.py（单机 TextMAS）+ 结果 jsonl
transport/     adapter.py（Transport 接口：Local/TCP/Lab 预留）
               server_b.py / client_a.py（双进程 TCP 跨机对话）
telemetry/     logger.py + runs.jsonl（推理/传输/负载统一日志）
eval/          dataset.py（50 题：自编 30 + GSM8K 20）
               check_answer.py（严格/宽松数字判定 + 单测）
               compare.py（TOKEN vs KV 同题对比）
               kv_equiv.py / delta_kv.py（KV 等价性 / Delta KV 原型）
demo/          build_report.py（静态报告）/ app.py（Gradio 工作台）
docs/          服务器部署清单.md / 实验室对接需求.md
```

## 实验流程（对应 steps.md）

| 步骤 | 内容 | 入口 |
|---|---|---|
| 1-2 | 选模型 + 单机 TextMAS | `agents/run_textmas.py` |
| 3 | 通信接口解耦（TOKEN/KV） | `transport/adapter.py` |
| 4 | 双进程 TCP 跨机文本对话 | `transport/server_b.py` + `client_a.py` |
| 5 | 监控日志 | `telemetry/logger.py` |
| 6 | KV Cache 捕获传输恢复 | `eval/kv_equiv.py` |
| 7 | TOKEN vs KV 对比 | `eval/compare.py` |
| 8 | 可视化 Demo | `demo/` |
| 9 | 接实验室通信系统（预留 `LabTransport`） | 见 `docs/实验室对接需求.md` |
| 10 | Delta KV 增量同步原型 | `eval/delta_kv.py` |

## 主要实验结果（Qwen3-0.6B，本机 CPU）

50 题（自编 30：算术/应用题/比例/逻辑/单位换算 ×6；GSM8K 标准子集 20）：

| 模式 | 严格通过 | 宽松通过 | 平均时延 | 平均传输量 |
|---|---|---|---|---|
| TOKEN（传文本） | 22/50 (44%) | 31/50 | 21.6s | 344B |
| KV（传 past_key_values） | 22/50 (44%) | 31/50 | 22.0s | 15.1MB |

**关键结论**：修正停止策略（KV 端同样生成到 EOS）后，两模式 50 题**逐题结果
完全一致**——证明 KV 传输无损、与直推等价；正确率只取决于模型能力。
0.6B 在 GSM8K 上 3/20 符合该量级公开基准，4B 终测数据待服务器重跑。

Delta KV（Step10）：只传新增位置的 KV，压缩比随 prompt 长度 3.1x→50x，
5 档长度等价性全部通过（`eval/delta_kv_table.md`）。

提示词实验发现：CoT 长链引导在 0.6B 上反而使严格通过率 42%→28%
（小模型撑不起长链推理）；"简短解答 + 末尾只给数字"最优。

## 注意事项

- 本机验证用 `Qwen/Qwen3-0.6B`；服务器阶段切 `Qwen/Qwen3-4B`（只改各脚本顶部 `MODEL`）
- `transformers>=5` 返回 `DynamicCache` 新格式，`DynamicCache.update` 为原地修改——
  做 KV 切分/拼接前必须 `deepcopy`（见 `eval/delta_kv.py`）
- 时延数据为 CPU 实测，仅用于同机相对对比，GPU 绝对值以服务器为准
