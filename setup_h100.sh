#!/bin/bash
# PNEUMA B200 PROTOCOL - FINAL PAYLOAD
# Automates Env Setup, Compilation, Dataset Download, and Model Standardization.

set -e

echo "🚀 INITIALIZING PNEUMA ON BLACKWELL (B200)..."

# 1. System Updates
echo "🛠️ Updating System..."
DEBIAN_FRONTEND=noninteractive apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential cmake python3-pip unzip

# 2. Python Dependencies
echo "🔥 Installing Python Libs..."
pip install --upgrade pip
pip install pandas openpyxl pypdf gdown huggingface_hub

# 3. VERIFY & COMPILE ENGINE (CUDA 12.8+)
echo "🔍 Checking NVCC..."
if ! command -v nvcc &> /dev/null; then
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        echo "✅ Found NVCC."
    else
        echo "⚠️ Installing CUDA Toolkit (Backup method)..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit
    fi
fi

echo "⚡ Compiling CUDA Engine..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# 4. DOWNLOAD DATASET (Google Drive)
# ID extracted from your link: 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc
echo "⬇️ Downloading Dataset..."
if [ ! -d "dataset" ]; then
    gdown --id 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc -O dataset.zip
    unzip -q dataset.zip -d dataset/
    rm dataset.zip
    echo "✅ Dataset Extracted."
else
    echo "✅ Dataset already exists."
fi

# 5. DOWNLOAD & RENAME MODELS
# This script downloads the specific repos you verified and renames them
# to 'lfm.gguf', 'qwen3b.gguf', 'phi4.gguf' for the benchmark scripts.
echo "📝 Running Model Harvester..."
cat <<EOF > download_models.py
from huggingface_hub import hf_hub_download
import os
import shutil

LOCAL_DIR = "models/"
os.makedirs(LOCAL_DIR, exist_ok=True)

# CONFIG: Repo -> Source File -> Local Destination Name
TARGETS = [
    {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "src": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "dest": "qwen3b.gguf"
    },
    {
        "repo": "itlwas/phi-4-Q4_K_M-GGUF",
        "src": "phi-4-q4_k_m.gguf",
        "dest": "phi4.gguf"
    },
    {
        "repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "src": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        "dest": "lfm.gguf"
    }
]

print(f"--- 🚀 DOWNLOADING MODELS ---")

for t in TARGETS:
    print(f"\n📥 Fetching: {t['repo']}...")
    try:
        # Download to cache/local dir
        file_path = hf_hub_download(
            repo_id=t['repo'],
            filename=t['src'],
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )

        # Rename to the standardized name expected by benchmark scripts
        final_path = os.path.join(LOCAL_DIR, t['dest'])
        os.rename(file_path, final_path)

        print(f"✅ Ready: {t['dest']}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)
EOF

python3 download_models.py

echo "✅ B200 ENVIRONMENT READY."
echo "👉 RUN SEQUENCE:"
echo "   1. python3 src/h100_lfm_warm.py"
echo "   2. python3 src/h100_lfm_cold.py"
echo "   3. python3 src/h100_transformer_warm.py"
echo "   4. python3 src/h100_transformer_cold.py"