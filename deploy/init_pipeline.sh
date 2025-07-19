#!/bin/bash

set -e
set -o pipefail

echo ""
echo "🧠 [DeFi Sniper AI Init] Launching high-fidelity pipeline initialization..."

# ---------- CONFIGURATION ----------
ENV_NAME="defi_sniper_env"
PYTHON_VERSION="3.10"
VENV_DIR="./.venv"
LOG_DIR="./logs"
DATA_DIR="./data"
CHARTS_DIR="./charts"
NOTEBOOK_ENTRY="run_pipeline.ipynb"
REQUIREMENTS_FILE="requirements.txt"
REQUIRED_COMMANDS=("python$PYTHON_VERSION" "pip" "git" "jupyter")

echo "🧬 Target Python: $PYTHON_VERSION"
echo "📁 Environment Path: $VENV_DIR"
echo ""

# ---------- VALIDATE HOST SYSTEM ----------
echo "🔎 Verifying system environment..."

for cmd in "${REQUIRED_COMMANDS[@]}"; do
  if ! command -v $cmd &>/dev/null; then
    echo "❌ Missing required command: $cmd"
    exit 1
  fi
done

echo "✅ All required CLI tools present."

# ---------- GPU CHECK ----------
echo ""
echo "🔍 Detecting GPU availability..."
GPU_INFO=$(python$PYTHON_VERSION -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))")
if [[ "$GPU_INFO" == *"GPU"* ]]; then
  echo "🚀 GPU FOUND: TensorFlow will use available GPU(s)"
else
  echo "⚠️ No GPU detected or TensorFlow not installed correctly. Expect latency."
fi

# ---------- MODULE HASH VALIDATION ----------
echo ""
echo "🔐 Hashing Python modules for integrity check..."
find . -maxdepth 1 -name "*.py" -exec sha256sum {} \; | tee $LOG_DIR/module_hashes_$(date +'%Y%m%d_%H%M%S').log

# ---------- CREATE LOGGING DIRS ----------
echo ""
echo "📂 Creating directory structure..."
mkdir -p "$VENV_DIR" "$LOG_DIR" "$DATA_DIR" "$CHARTS_DIR"

# ---------- CREATE VENV ----------
echo ""
echo "🐍 Creating Python $PYTHON_VERSION virtual environment..."
python$PYTHON_VERSION -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo "✅ Virtual environment activated."

# ---------- PIP INSTALL ----------
echo ""
echo "📦 Installing dependencies from $REQUIREMENTS_FILE..."
pip install --upgrade pip wheel
pip install -r $REQUIREMENTS_FILE | tee $LOG_DIR/pip_install_$(date +'%Y%m%d_%H%M%S').log

# ---------- IPYKERNEL SETUP ----------
echo ""
echo "🧠 Registering Jupyter kernel..."
python -m ipykernel install --user --name=$ENV_NAME --display-name "DeFi Sniper AI"

# ---------- SYNTAX VALIDATION ----------
echo ""
echo "🔍 Validating all Python modules..."
ERROR_FLAG=0
for file in *.py; do
  if ! python -m py_compile "$file"; then
    echo "❌ Compilation failed: $file"
    ERROR_FLAG=1
  fi
done

if [ $ERROR_FLAG -eq 1 ]; then
  echo "❌ Module validation failed. Fix errors and rerun."
  exit 1
fi
echo "✅ All modules passed syntax validation."

# ---------- GIT CONTEXT ----------
echo ""
echo "🔗 Git Context:"
echo "    Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "    Latest Commit: $(git log -1 --pretty=format:'%h - %s (%ci)')"

# ---------- ENVIRONMENT SNAPSHOT ----------
echo ""
echo "📸 Capturing full environment snapshot..."
pip freeze > $LOG_DIR/env_snapshot_$(date +'%Y%m%d_%H%M%S').txt

# ---------- SYSTEM METRICS ----------
echo ""
echo "🧾 System Information:"
echo "    Hostname: $(hostname)"
echo "    OS: $(uname -a)"
echo "    CPU: $(lscpu | grep 'Model name' | uniq | awk -F ':' '{print $2}')"
echo "    Mem: $(free -h | grep Mem | awk '{print $2}')"
echo "    Disk: $(df -h . | tail -1 | awk '{print $4}')"
echo ""

# ---------- FINAL OUTPUT ----------
echo ""
echo "🚀 Pipeline environment initialized."
echo "To launch:"
echo ""
echo "    source $VENV_DIR/bin/activate"
echo "    jupyter notebook $NOTEBOOK_ENTRY"
echo ""
echo "🧠 Kernel registered as: $ENV_NAME"
echo "📄 Logs saved to: $LOG_DIR"
echo ""

exit 0
