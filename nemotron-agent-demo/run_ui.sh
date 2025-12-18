#!/usr/bin/env bash
set -euo pipefail

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=${PORT:-7860}
PYTHONPATH=src:${PYTHONPATH:-} python -m src.ui.app
