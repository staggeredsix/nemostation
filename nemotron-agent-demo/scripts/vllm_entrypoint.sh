#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"

export HF_HOME

if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

ensure_hf_deps() {
  python - <<'PY'
import importlib
missing = []
for name in ("huggingface_hub", "transformers"):
    if importlib.util.find_spec(name) is None:
        missing.append(name)
if missing:
    raise SystemExit("missing")
PY
}

if ! ensure_hf_deps; then
  echo "Installing Hugging Face dependencies..." >&2
  pip install --no-cache-dir huggingface_hub transformers
fi

echo "Warming model cache…"
python - <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
hf_home = os.environ.get("HF_HOME")

try:
    snapshot_download(
        repo_id=model_id,
        token=token,
        cache_dir=hf_home,
        local_files_only=False,
        resume_download=True,
        allow_patterns=[
            "*.safetensors",
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "tokenizer.*",
            "vocab.*",
            "merges.txt",
        ],
    )
except Exception as exc:
    print(
        f"ERROR: Failed to warm model cache for {model_id}. "
        "Check HF_TOKEN for gated access and ensure the cache path is writable.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc
PY
echo "Warm complete"

exec python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model "${MODEL_ID}" \
  --trust-remote-code \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
