#!/usr/bin/env bash
# ConfliBERT-v2 environment setup for WSL2 Ubuntu-22.04 (run as root).
# Creates an ext4-local venv (fast) at /root/cb2-venv. Code stays on /mnt/f.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== apt essentials ==="
apt-get update -y
apt-get install -y --no-install-recommends \
  build-essential git curl ca-certificates \
  python3-venv python3-dev pkg-config

echo "=== install uv (fast pip) ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "=== create venv (Python 3.11, ext4-local) ==="
uv venv --python 3.11 /root/cb2-venv
# shellcheck disable=SC1091
source /root/cb2-venv/bin/activate
uv pip install --upgrade pip wheel setuptools

echo "=== PyTorch (CUDA 12.4) ==="
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== HF + data stack ==="
uv pip install \
  "transformers>=4.52.2,<5" "datasets>=2.19" accelerate tokenizers evaluate \
  scikit-learn pandas pyarrow orjson tqdm safetensors sentencepiece \
  huggingface_hub

echo "=== FlashAttention 2 ==="
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 uv pip install "flash-attn==2.6.3" --no-build-isolation --no-cache

echo "=== verify ==="
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda avail", torch.cuda.is_available(), "| cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
PY
echo "=== DONE ==="
