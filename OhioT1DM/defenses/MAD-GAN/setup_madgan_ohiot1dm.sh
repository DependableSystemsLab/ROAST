#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MADGAN_DIR="$SCRIPT_DIR"
VENV_DIR="$MADGAN_DIR/venv_madgan"
MADGAN_USE_CUDA="${MADGAN_USE_CUDA:-0}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_python39() {
  if [ -n "${MADGAN_PYTHON:-}" ] && [ -x "${MADGAN_PYTHON}" ]; then
    echo "${MADGAN_PYTHON}"
    return 0
  fi

  if command -v python3.9 >/dev/null 2>&1; then
    command -v python3.9
    return 0
  fi

  if command -v pyenv >/dev/null 2>&1; then
    local pyenv_root
    pyenv_root="$(pyenv root 2>/dev/null || true)"
    if [ -n "$pyenv_root" ] && [ -x "$pyenv_root/versions/3.9.19/bin/python3.9" ]; then
      echo "$pyenv_root/versions/3.9.19/bin/python3.9"
      return 0
    fi
  fi

  return 1
}

if [ ! -d "$MADGAN_DIR" ]; then
  echo ">>> MAD-GAN directory not found: $MADGAN_DIR"
  exit 1
fi

PYTHON_BIN="$(resolve_python39 || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo ">>> Python 3.9 is required for this MAD-GAN environment."
  echo ">>> Install Python 3.9 (or set MADGAN_PYTHON=/absolute/path/to/python3.9) and rerun."
  exit 1
fi

echo ">>> Creating isolated OhioT1DM MAD-GAN environment"
VENV_PYTHON="$VENV_DIR/bin/python3.9"
if [ ! -x "$VENV_PYTHON" ] || ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo ">>> Reusing existing OhioT1DM MAD-GAN environment"
fi

echo ">>> Bootstrapping pip/setuptools/wheel for Python 3.9"
"$VENV_PYTHON" -m ensurepip --upgrade || true
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel || true

echo ">>> Installing OhioT1DM MAD-GAN requirements"
if ! "$VENV_PYTHON" -m pip install --no-cache-dir --prefer-binary -r "$MADGAN_DIR/requirements.txt"; then
  echo ">>> Bulk install failed; retrying requirements one by one for better resilience"
  while IFS= read -r dep; do
    dep="${dep%%#*}"
    dep="$(echo "$dep" | xargs)"
    if [ -z "$dep" ]; then
      continue
    fi
    echo ">>> Installing $dep"
    "$VENV_PYTHON" -m pip install --no-cache-dir --prefer-binary "$dep"
  done < "$MADGAN_DIR/requirements.txt"
fi

if is_truthy "$MADGAN_USE_CUDA"; then
  if [ "$(uname -s)" != "Linux" ]; then
    echo ">>> CUDA mode requested, but CUDA-enabled TensorFlow wheels are supported on Linux in this flow."
    echo ">>> Falling back to CPU-only TensorFlow install."
  elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ">>> CUDA mode requested, but nvidia-smi was not found."
    echo ">>> Ensure NVIDIA driver/CUDA runtime are installed, then rerun."
    echo ">>> Continuing anyway and attempting tensorflow[and-cuda] install."
  fi

  echo ">>> Installing TensorFlow with CUDA extras"
  "$VENV_PYTHON" -m pip install --no-cache-dir --upgrade "tensorflow[and-cuda]"
else
  echo ">>> CPU mode selected (set MADGAN_USE_CUDA=1 to enable CUDA install path)"
fi

echo ">>> TensorFlow device check"
"$VENV_PYTHON" - <<'PY'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"Detected GPUs: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU[{i}]: {gpu}")
PY

echo ">>> OhioT1DM MAD-GAN setup complete"
echo ">>> Activate with: source $VENV_DIR/bin/activate"
