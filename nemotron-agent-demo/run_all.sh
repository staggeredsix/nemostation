#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMPOSE_ARGS=(-f "${ROOT_DIR}/docker-compose.yml")
if [[ "${VLLM_MODE:-compose}" != "host" ]]; then
  COMPOSE_ARGS+=(-f "${ROOT_DIR}/docker-compose.nemotron3.yml")
fi

docker compose "${COMPOSE_ARGS[@]}" up --build "$@"
