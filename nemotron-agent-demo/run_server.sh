#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}
PORT=${PORT:-8000}
HOST=0.0.0.0

python -m vllm.entrypoints.openai.api_server \
  --host ${HOST} \
  --port ${PORT} \
  --model "${MODEL_ID}" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  "$@" &
SERVER_PID=$!

cleanup() {
  kill ${SERVER_PID} 2>/dev/null || true
}
trap cleanup INT TERM

PYTHONPATH=src:${PYTHONPATH:-} python -m src.server.healthcheck --host ${HOST} --port ${PORT}

echo "SERVER READY"
wait ${SERVER_PID}
