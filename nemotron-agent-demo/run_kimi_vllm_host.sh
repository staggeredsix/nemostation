#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="${SCRIPT_DIR}/.venv-kimi-vllm"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --upgrade pip
  python -m pip install uv
fi

uv pip install vllm --pre \
  --extra-index-url https://wheels.vllm.ai/1d495c2f92c7e75f580c2f4b465823f2fb688abe/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
uv pip install model-hosting-container-standards

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

export HF_HOME=/mnt/raid/kimik2/hf
export HUGGINGFACE_HUB_CACHE=/mnt/raid/kimik2/hf
export VLLM_CACHE_ROOT=/mnt/raid/kimik2/vllm
export TMPDIR=/mnt/raid/kimik2/tmp

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${VLLM_CACHE_ROOT}" "${TMPDIR}"

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_CPU_OFFLOAD_GB=${VLLM_CPU_OFFLOAD_GB:-360}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}

vllm serve /mnt/raid/kimik2/hf/Kimi-K2-Thinking-NVFP4 \
  --served-model-name kimi-k2-nvfp4 \
  --trust-remote-code \
  --host 0.0.0.0 --port "${VLLM_PORT}" \
  --cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
