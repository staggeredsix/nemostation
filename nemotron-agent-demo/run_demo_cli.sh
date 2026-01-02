#!/usr/bin/env bash
set -euo pipefail

VLLM_BASE_URL=${VLLM_BASE_URL:-http://localhost:8000/v1}

python - <<PY
import os
import sys
import requests

base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
url = f"{base_url}/models"
try:
    resp = requests.get(url, timeout=3)
    if resp.status_code == 200:
        sys.exit(0)
except requests.RequestException:
    pass
print(f"Server not ready at {url}. Start the vLLM server first.", file=sys.stderr)
sys.exit(1)
PY

PYTHONPATH=src:${PYTHONPATH:-} python -m src.demo.cli "$@"
