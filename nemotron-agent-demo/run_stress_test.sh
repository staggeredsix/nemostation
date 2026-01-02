#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src:${PYTHONPATH:-} python -m src.demo.stress_test "$@"
