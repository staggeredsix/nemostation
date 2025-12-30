#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="${SCRIPT_DIR}/.venv-kimi-vllm"

if [[ ! -d "${VENV_DIR}" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  elif command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN=python3.10
  else
    echo "ERROR: python3.11 or python3.10 is required to create the vLLM venv for cu130 wheels." >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

uv pip install vllm --pre --extra-index-url https://wheels.vllm.ai/1d495c2f92c7e75f580c2f4b465823f2fb688abe/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
uv pip install model-hosting-container-standards

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

export HF_HOME=/mnt/raid/kimik2/hf
export HUGGINGFACE_HUB_CACHE=/mnt/raid/kimik2/hf
export VLLM_CACHE_ROOT=/mnt/raid/kimik2/vllm
export TMPDIR=/mnt/raid/kimik2/tmp

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${VLLM_CACHE_ROOT}" "${TMPDIR}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
if not (torch.version.cuda and str(torch.version.cuda).startswith("13")):
    raise SystemExit(
        "ERROR: torch.version.cuda is not 13.x; cu130 wheels did not resolve (would lead to libcudart.so.12 errors). Use python3.10/3.11 venv."
    )
PY

vllm serve /mnt/raid/kimik2/hf/Kimi-K2-Thinking-NVFP4 --cpu-offload-gb 360 --trust-remote-code --host 0.0.0.0 --port 8000 --served-model-name kimi-k2-nvfp4
