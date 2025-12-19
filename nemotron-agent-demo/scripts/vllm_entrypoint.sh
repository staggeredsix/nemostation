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

pip uninstall -y importlib || true
pip install -U "vllm>=0.12.0" huggingface_hub transformers

echo "Warming model cache…"
snapshot_path="$(python - <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
hf_home = os.environ.get("HF_HOME")

try:
    snapshot_path = snapshot_download(
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
    print(snapshot_path)
except Exception as exc:
    print(
        f"ERROR: Failed to warm model cache for {model_id}. "
        "Check HF_TOKEN for gated access and ensure the cache path is writable.",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc
PY
)"
echo "Warm complete"

mkdir -p /workspace
wget -O /workspace/nano_v3_reasoning_parser.py \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py

exec vllm serve --model "${MODEL_ID}" \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 8000 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /workspace/nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3
