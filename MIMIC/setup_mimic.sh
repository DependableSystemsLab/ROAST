#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
VENV_DIR="$BASE_DIR/venv_mimic"
PYTHON_BIN="${MIMIC_PYTHON:-python3.7}"

echo ">>> Creating MIMIC environment"
VENV_PYTHON="$VENV_DIR/bin/python3.7"
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo ">>> Reusing existing MIMIC environment"
fi

echo ">>> Upgrading pip toolchain"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo ">>> Installing requirements"
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo ">>> Installing PyTorch"
"$VENV_PYTHON" -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

echo ">>> Copying MIMIC data"
cp -r ~/Downloads/MIMIC/original/* "$BASE_DIR/" || true

echo ">>> Installing URET"
cd "$SCRIPT_DIR/URET"
"$VENV_PYTHON" -m pip install -e .
cd "$BASE_DIR"

echo ">>> MIMIC setup complete"
