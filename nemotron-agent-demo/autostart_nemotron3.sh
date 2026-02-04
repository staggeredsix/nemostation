#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set. If the model is gated, export HF_TOKEN before the first run." >&2
fi

docker compose \
  -f "${ROOT_DIR}/docker-compose.yml" \
  -f "${ROOT_DIR}/docker-compose.nemotron3.yml" \
  up -d --build

docker compose \
  -f "${ROOT_DIR}/docker-compose.yml" \
  -f "${ROOT_DIR}/docker-compose.nemotron3.yml" \
  ps
