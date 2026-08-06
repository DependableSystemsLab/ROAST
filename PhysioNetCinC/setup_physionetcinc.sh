#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR"
VENV_DIR="$BASE_DIR/venv_physionetcinc"
PYTHON_BIN="${PHYSIONETCINC_PYTHON:-python3.9}"

echo ">>> Creating PhysioNetCinc environment"
VENV_PYTHON="$VENV_DIR/bin/python3.9"
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo ">>> Reusing existing PhysioNetCinc environment"
fi

echo ">>> Upgrading pip toolchain"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo ">>> Installing requirements"
"$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo ">>> Copying PhysioNetCinc data"
rm -rf "$BASE_DIR/input" || true
cp -r ~/Downloads/Sepsis/files/challenge-2019/1.0.0/training "$BASE_DIR/input" || true

echo ">>> Installing URET"
cd "$SCRIPT_DIR/URET"
"$VENV_PYTHON" -m pip install -e .
cd "$BASE_DIR"

echo ">>> PhysioNetCinc setup complete"
