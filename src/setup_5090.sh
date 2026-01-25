#!/bin/bash
# setup_5090.sh - Optimized for RTX 5090 (Blackwell sm_120)
# Template: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

echo "===================================================================="
echo "🚀 PROJECT PNEUMA: RTX 5090 BLACKWELL FULL ACTIVATION"
echo "===================================================================="

# 1. Branch Setup
if [ ! -d "ProjectPNEUMA" ]; then
    git clone --single-branch --branch rtx5090bench https://github.com/taurussilver24/ProjectPNEUMA.git
    cd ProjectPNEUMA
else
    cd ProjectPNEUMA
    git pull origin rtx5090bench
fi

# 2. Build Tools
apt-get update && apt-get install -y build-essential cmake git wget unzip zip nvidia-cuda-toolkit

# 3. Optimized sm_120 Compilation
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DGGML_FLASH_ATTN=on -DCMAKE_CUDA_ARCHITECTURES=120"
export FORCE_CMAKE=1

echo "📦 Compiling llama-cpp-python for Blackwell..."
pip uninstall -y llama-cpp-python
pip install llama-cpp-python --no-cache-dir

# 4. STRICT DUAL-GPU READINESS CHECK (Exit on failure)
echo "🔍 PERFORMING STAGE-GATE GPU CHECKS..."
python3 <<EOF
import sys
from llama_cpp import llama_supports_gpu_offload
if not llama_supports_gpu_offload():
    print("❌ CRITICAL: llama-cpp-python failed to compile with CUDA hooks.")
    sys.exit(1)
print("✅ Llama-CPP CUDA Backend Verified.")
EOF
if [ $? -ne 0 ]; then echo "🛑 ABORTING: GPU check failed."; exit 1; fi

# 5. Model/Data Ingestion
echo "📥 Ingesting Models and Data..."
mkdir -p models logs data/input
wget -nc -O models/lfm-2.5-1.2b.Q4_K_M.gguf "https://huggingface.co/bartowski/LFM-2.5-1.2B-GGUF/resolve/main/LFM-2.5-1.2B-Q4_K_M.gguf"
wget -nc -O models/qwen2.5-3b-instruct.Q4_K_M.gguf "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
wget -nc -O models/phi-4.Q4_K_M.gguf "https://huggingface.co/bartowski/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf"
if [ -f "../dataset.zip" ]; then unzip -n ../dataset.zip -d data/input/; fi

# 6. SEQUENTIAL MARATHON EXECUTION
echo "🏃 STARTING THE 4-VARIANT MARATHON..."
python3 src/lfm_warm.py
python3 src/lfm_cold.py
python3 src/transformer_warm.py
python3 src/transformer_cold.py

# 7. LOG PACKAGING & DOWNLOAD
echo "📦 PACKAGING RESULTS..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
zip -r "PNEUMA_5090_RESULTS_${TIMESTAMP}.zip" logs/*.csv

echo "===================================================================="
echo "✅ ALL BENCHMARKS COMPLETE."
echo "🎁 To download your results, run this command:"
runpodctl send "PNEUMA_5090_RESULTS_${TIMESTAMP}.zip"
echo "===================================================================="