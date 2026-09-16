#!/usr/bin/env bash
# One-time environment setup on an NCSA Delta LOGIN node.
#
#   cp hpc/paths.env.example hpc/paths.env   # edit CB2_ALLOC first!
#   bash hpc/setup_delta.sh
#
# Creates the venv at $CB2_VENV (on /projects, persistent), installs the pinned
# dependency set, pre-downloads the base model + tokenizer into $HF_HOME so
# compute jobs never depend on the Hub being reachable, and runs a smoke test.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f hpc/paths.env ] || { echo "ABORT: copy hpc/paths.env.example to hpc/paths.env and edit it first"; exit 1; }
source hpc/paths.env
[ "$CB2_ALLOC" != "CHANGE_ME" ] || { echo "ABORT: set CB2_ALLOC in hpc/paths.env"; exit 1; }

echo "=== modules ==="
module reset
module load python/3.11 2>/dev/null || module load anaconda3_gpu
python3 --version

echo "=== venv at $CB2_VENV ==="
mkdir -p "$(dirname "$CB2_VENV")" "$CB2_SCRATCH" "$HF_HOME"
python3 -m venv "$CB2_VENV"
source "$CB2_VENV/bin/activate"
pip install --upgrade pip wheel setuptools

echo "=== PyTorch (CUDA 12.4 wheels; Delta A100/A40 drivers are compatible) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== project dependencies ==="
pip install -r requirements.txt

echo "=== FlashAttention 2 (installed after torch) ==="
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 pip install "flash-attn==2.6.3" --no-build-isolation --no-cache-dir

echo "=== pre-download base models + tokenizer into HF_HOME (both sizes) ==="
python - <<'PY'
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
for name in ("answerdotai/ModernBERT-base", "answerdotai/ModernBERT-large"):
    AutoTokenizer.from_pretrained(name)
    AutoModelForMaskedLM.from_pretrained(name)
    print("cached:", name, "->", os.environ["HF_HOME"])
PY

echo "=== smoke test (CPU-only on the login node; GPU is checked in-job) ==="
python env/smoke_test.py || true

echo "=== DONE. Next: sbatch hpc/pack_corpus.sbatch, then sbatch hpc/pretrain.sbatch ==="
