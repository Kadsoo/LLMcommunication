#!/usr/bin/env bash
# 一键环境初始化 (Ubuntu): 建venv -> 装依赖 -> 验GPU -> 下模型 -> 冒烟.
# 用法: bash scripts/setup.sh [--model Qwen/Qwen3-4B] [--4bit]
# 可重入: 已存在的内容自动跳过. 任一步失败即停并打印原因.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="Qwen/Qwen3-4B"
BIT4=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model=*) MODEL="${1#--model=}"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --4bit) BIT4="1"; shift ;;
    *) echo "[setup] 未知参数: $1" >&2; exit 1 ;;
  esac
done

fail() { echo "[setup] FAILED: $1" >&2; exit 1; }
ok()   { echo "[setup] OK: $1"; }

command -v python3 >/dev/null || fail "未找到 python3, 先装 Python 3.10+"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "[setup] WARN: 无 nvidia-smi, 将使用 CPU(很慢, 仅联调)"
[ "$(python3 -c 'import sys; print(sys.version_info >= (3,10))')" = "True" ] || fail "Python 版本过低, 需要 3.10+"

if [ ! -d .venv ]; then
  python3 -m venv .venv || fail "创建 venv 失败(缺 python3-venv 就 apt install python3-venv)"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "venv 已就绪: $(python -c 'import sys; print(sys.version.split()[0])')"

CUDA_URL="https://download.pytorch.org/whl/cu121"
python -c "import torch" 2>/dev/null || pip install --upgrade pip torch --index-url "$CUDA_URL" || fail "安装 torch 失败"
pip install transformers accelerate safetensors pyarrow gradio pyyaml huggingface_hub bitsandbytes || fail "安装依赖失败"
ok "依赖已安装"

python -c "import torch; print('[setup] torch', torch.__version__, 'cuda=', torch.cuda.is_available())"
if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  VRAM=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // 1024**3)")
  echo "[setup] GPU 显存约 ${VRAM}GB"
  if [ "$VRAM" -lt 16 ] && [ -z "$BIT4" ]; then
    echo "[setup] WARN: 显存 <16GB, 运行时将自动 4bit 量化; 也可重跑 bash scripts/setup.sh --4bit 强制"
  fi
fi

export LLM_MODEL="$MODEL"
export LLM_4BIT="$([ -n "$BIT4" ] && echo 1 || echo 0)"
echo "[setup] 预下载权重 $LLM_MODEL ..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || fail "权重下载失败(检查网络/代理/HF连通性)"
ok "权重已缓存"

echo "[setup] 冒烟测试..."
python scripts/smoke.py || fail "冒烟失败, 看上方报错"
ok "全部完成. 评测: LLM_MODEL=$MODEL bash scripts/run_eval.sh"
