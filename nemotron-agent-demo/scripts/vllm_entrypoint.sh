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

config_path="${snapshot_path}/config.json"
if [[ -f "${config_path}" ]]; then
  echo "Inspecting model config at ${config_path}"
  CONFIG_PATH="${config_path}" python - <<'PY'
import json
import os

config_path = os.environ["CONFIG_PATH"]
with open(config_path, "r", encoding="utf-8") as handle:
    config = json.load(handle)

for key in sorted(config.keys()):
    if "norm" in key or "eps" in key:
        print(f"config[{key}] = {config[key]!r}")
PY
else
  echo "WARNING: config.json not found at ${config_path}" >&2
fi

if [[ -z "${RMS_NORM_EPS:-}" ]]; then
  if [[ -f "${config_path}" ]]; then
    RMS_NORM_EPS="$(
      CONFIG_PATH="${config_path}" python - <<'PY'
import json
import os

config_path = os.environ["CONFIG_PATH"]
with open(config_path, "r", encoding="utf-8") as handle:
    config = json.load(handle)

if "rms_norm_eps" in config:
    value = config["rms_norm_eps"]
elif "rms_norm_epsilon" in config and "rms_norm_eps" not in config:
    value = config["rms_norm_epsilon"]
else:
    value = 1e-6

print(value)
PY
    )"
  else
    RMS_NORM_EPS="1e-6"
  fi
fi

export RMS_NORM_EPS
HF_OVERRIDES="$(printf '{"rms_norm_eps": %s}' "${RMS_NORM_EPS}")"

exec python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model "${MODEL_ID}" \
  --trust-remote-code \
  --hf-overrides "${HF_OVERRIDES}" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90
