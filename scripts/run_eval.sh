#!/usr/bin/env bash
# 一键评测: 50题对比 + Delta KV + 重建报告页. 日志落 logs/.
# 用法: LLM_MODEL=Qwen/Qwen3-4B bash scripts/run_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen3-4B}"
mkdir -p logs
TS=$(date +%Y%m%d-%H%M%S)

echo "[eval] 模型: $LLM_MODEL"
echo "[eval] (1/3) 50题 TOKEN vs KV 对比..."
python eval/compare.py 2>&1 | tee "logs/compare-$TS.log"
echo "[eval] (2/3) Delta KV 压缩比..."
python eval/delta_kv.py 2>&1 | tee "logs/delta-$TS.log"
echo "[eval] (3/3) 重建报告页..."
python demo/build_report.py 2>&1 | tee "logs/report-$TS.log"
echo "[eval] DONE. 报告: demo/index.html  日志: logs/"
