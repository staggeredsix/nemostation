#!/usr/bin/env bash
set -euo pipefail

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=${PORT:-7860}
export VLLM_BASE_URL=${VLLM_BASE_URL:-http://host.docker.internal:8000/v1}
export VLLM_MODEL_ID=${VLLM_MODEL_ID:-mistral/Mistral-Large-3-675B-Instruct-2512-NVFP4/}
export VLLM_TIMEOUT_S=${VLLM_TIMEOUT_S:-120}
PYTHONPATH=src:${PYTHONPATH:-} python -m src.ui.app
