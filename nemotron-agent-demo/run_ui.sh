#!/usr/bin/env bash
set -euo pipefail

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=${PORT:-7860}
export VLLM_BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}
PYTHONPATH=src:${PYTHONPATH:-} python -m src.ui.app
