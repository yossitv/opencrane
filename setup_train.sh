#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo
echo "Training environment is ready."
echo "Activate it with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Optional:"
echo "  cp \"$ROOT_DIR/.env.example\" \"$ROOT_DIR/.env\""
echo
echo "Start training with:"
echo "  python \"$ROOT_DIR/training/train.py\""
