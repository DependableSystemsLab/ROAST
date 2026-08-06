#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
VENV_DIR="$BASE_DIR/venv_ohiot1dm"
PYTHON_BIN="${OHIOT1DM_PYTHON:-python3.9}"

echo ">>> Creating OhioT1DM environment"
VENV_PYTHON="$VENV_DIR/bin/python3.9"
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo ">>> Reusing existing OhioT1DM environment"
fi

echo ">>> Upgrading pip toolchain"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo ">>> Installing requirements"
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo ">>> Copying OhioT1DM data"
mkdir -p "$BASE_DIR/data"
mkdir -p "$BASE_DIR/data/raw"
cp -r ~/Downloads/OhioT1DM/raw_data/. "$BASE_DIR/data/raw/" || true    # Important: Replace ~/Downloads/OhioT1DM/raw_data with the actual path to the raw data
# mkdir -p "$BASE_DIR/data/processed"
# cp -r ~/Downloads/OhioT1DM/processed_data/2020data "$BASE_DIR/data/processed/" || true
# cp -r ~/Downloads/OhioT1DM/processed_data/2018data "$BASE_DIR/data/processed/" || true

# echo ">>> Copying pretrained models"
# cp -r ~/Downloads/OhioT1DM/models/PRETRAINS "$BASE_DIR/" || true
echo ">>> Downloading pretrained models"
if [ -d "$BASE_DIR/PRETRAINS" ]; then
  echo ">>> Pretrained models already present; skipping download"
else
  "$VENV_PYTHON" -m pip install gdown
  if [ ! -f "$BASE_DIR/PRETRAINS.zip" ]; then
    "$VENV_PYTHON" -m gdown 1VXO2wT7M0htjVGbd6hk2935q52AteI7A -O "$BASE_DIR/PRETRAINS.zip"
  fi
  unzip -o "$BASE_DIR/PRETRAINS.zip" -d "$BASE_DIR/"
fi

echo ">>> Installing URET"
cd "$SCRIPT_DIR/URET"
"$VENV_PYTHON" -m pip install -e .
cd "$BASE_DIR"

echo ">>> OhioT1DM setup complete"
