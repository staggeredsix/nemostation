#!/usr/bin/env bash
set -euo pipefail

if ! python -m src.server.healthcheck --host localhost --port ${PORT:-8000}; then
  echo "Server not ready at http://localhost:${PORT:-8000}. Start the server first." >&2
  exit 1
fi

PYTHONPATH=src:${PYTHONPATH:-} python -m src.demo.cli "$@"
