#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="python3"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python3 is required but was not found on PATH" >&2
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

pushd "${PROJECT_ROOT}" >/dev/null
python -m pip install -e ".[server,tokenizer,embeddings,faiss,mcp,dev]"
popd >/dev/null

if [[ "${1:-}" != "--skip-tests" ]]; then
    echo "Running test suite"
    python -m pytest
fi

echo
echo "Deployment complete!"
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
