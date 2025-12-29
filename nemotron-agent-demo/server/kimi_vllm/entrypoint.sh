#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/raid/kimik2/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/mnt/raid/kimik2/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/mnt/raid/kimik2/vllm}"
export TMPDIR="${TMPDIR:-/mnt/raid/kimik2/tmp}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$VLLM_CACHE_ROOT" "$TMPDIR"

KIMI_MODEL_PATH="${KIMI_MODEL_PATH:-/mnt/raid/kimik2/hf/Kimi-K2-Thinking-NVFP4}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
VLLM_SERVED_MODEL_NAME="kimi-k2-nvfp4"

VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")

cat <<EOF
Starting vLLM ${VLLM_VERSION}
Model: ${KIMI_MODEL_PATH}
Limits: max_model_len=${VLLM_MAX_MODEL_LEN}, max_num_seqs=${VLLM_MAX_NUM_SEQS}, max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}
KV cache dtype: ${VLLM_KV_CACHE_DTYPE}
GPU memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}
EOF

aexec=(
  python -m vllm.entrypoints.openai.api_server
  --model "$KIMI_MODEL_PATH"
  --served-model-name "$VLLM_SERVED_MODEL_NAME"
  --trust-remote-code
  --host 0.0.0.0
  --port 8000
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype "$VLLM_KV_CACHE_DTYPE"
  --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
)

exec "${aexec[@]}"
