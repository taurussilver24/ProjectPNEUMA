#!/bin/bash
# PNEUMA RTX 4090 PROTOCOL (The "Easy Mode")
# Hardware: NVIDIA RTX 4090 (Ada Lovelace)
# Strategy: Use Pre-built Wheels (No compilation required)

set -e

echo "===================================================================="
echo "🚀 INITIATING RTX 4090 BENCHMARK PROTOCOL"
echo "===================================================================="

# 1. CLEANUP (Just in case we are re-running on a dirty pod)
# We remove the old repo folder to ensure a clean clone
rm -rf build models data dataset.zip ProjectPNEUMA
pip uninstall -y llama-cpp-python

# 2. CLONE REPO
# We clone the specific branch to get the latest scripts
git clone --single-branch --branch rtx5090bench https://github.com/taurussilver24/ProjectPNEUMA.git
cd ProjectPNEUMA

# 3. INSTALL DEPENDENCIES
echo "🛠️ Installing Python Tools..."
apt-get update && apt-get install -y unzip
pip install pandas pypdf openpyxl gdown huggingface_hub

# 4. INSTALL LLAMA-CPP-PYTHON (PRE-BUILT)
# We use the cu121 wheel. It is compatible with almost all RunPod 4090 instances.
# NO COMPILATION NECESSARY.
echo "📦 Installing Llama-CPP Pre-built Wheel..."
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# 5. DOWNLOAD DATA (Robust GDown + Flatten)
echo "⛽ Downloading Dataset..."
mkdir -p data/input models logs
gdown --id 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc -O dataset.zip
unzip -o dataset.zip -d data/input/
# FIX: Flatten nested folder
mv data/input/dataset/*.pdf data/input/ 2>/dev/null || true
echo "✅ FUEL CHECK: $(ls data/input/*.pdf | wc -l) documents ready."

# 6. DOWNLOAD MODELS (Python Method)
echo "🧠 Downloading Models..."
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
files = [
    ('bartowski/LFM-2.5-1.2B-GGUF', 'LFM-2.5-1.2B-Q4_K_M.gguf'),
    ('bartowski/Qwen2.5-3B-Instruct-GGUF', 'Qwen2.5-3B-Instruct-Q4_K_M.gguf'),
    ('bartowski/phi-4-GGUF', 'phi-4-Q4_K_M.gguf')
]
for repo, filename in files:
    print(f'Downloading {filename}...')
    path = hf_hub_download(repo, filename)
    shutil.copy(path, f'models/{filename.lower()}')
"

# 7. FIRE THE MARATHON
echo "🔥 GPU ENGAGED. STARTING RUN..."
# Verify GPU visibility
python3 -c "from llama_cpp import Llama; print('✅ 4090 GPU DETECTED')"

python3 src/lfm_warm.py
python3 src/lfm_cold.py
python3 src/transformer_warm.py
python3 src/transformer_cold.py

# 8. EXFILTRATE
echo "📦 PACKAGING RESULTS..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
zip -r "PNEUMA_4090_FINAL_${TIMESTAMP}.zip" logs/*.csv

echo "===================================================================="
echo "🎁 SUCCESS. RUN THIS COMMAND LOCALLY:"
echo "runpodctl receive PNEUMA_4090_FINAL_${TIMESTAMP}.zip"
runpodctl send "PNEUMA_4090_FINAL_${TIMESTAMP}.zip"
echo "===================================================================="