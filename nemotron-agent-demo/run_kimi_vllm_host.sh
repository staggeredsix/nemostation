#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-kimi-vllm"

PYTHON_BIN=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
else
  echo "Error: python3.11 or python3.10 is required on this host (python3.12 is not supported)." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

uv pip install vllm --pre --extra-index-url https://wheels.vllm.ai/1d495c2f92c7e75f580c2f4b465823f2fb688abe/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv pip install model-hosting-container-standards

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

export HF_HOME="${REPO_ROOT}/_hf_cache"
export HUGGINGFACE_HUB_CACHE="${REPO_ROOT}/_hf_cache"
export VLLM_CACHE_ROOT="${REPO_ROOT}/_vllm_cache"
export TMPDIR="${REPO_ROOT}/_tmp"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${VLLM_CACHE_ROOT}" "${TMPDIR}"

MODEL_DIR="${HUGGINGFACE_HUB_CACHE}/hub/models--nvidia--Kimi-K2-Thinking-NVFP4"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Error: ${MODEL_DIR}/config.json not found. Ensure the model is present in the HF hub cache." >&2
  exit 1
fi

vllm serve "${MODEL_DIR}" \
  --cpu-offload-gb 360 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name kimi-k2-nvfp4
