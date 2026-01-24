#!/bin/bash
# PNEUMA B200 PROTOCOL (CUDA 12.8+)
# Optimized for runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

set -e

echo "🚀 INITIALIZING PNEUMA ON BLACKWELL (B200)..."

# 1. System Updates
echo "🛠️ Updating System..."
# Ubuntu 24.04 handles apt slightly differently, quiet mode helps prevent interaction
DEBIAN_FRONTEND=noninteractive apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential cmake python3-pip

# 2. Python Dependencies
echo "🔥 Installing Python Libs..."
pip install --upgrade pip
pip install pandas openpyxl pypdf

# 3. VERIFY CUDA COMPILER (NVCC)
echo "🔍 Checking for NVCC..."
if ! command -v nvcc &> /dev/null; then
    echo "⚠️ NVCC not found in PATH! Attempting to locate..."
    # Common location in RunPod images
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export PATH="/usr/local/cuda/bin:$PATH"
        echo "✅ Found NVCC at /usr/local/cuda/bin/nvcc"
    else
        echo "❌ NVCC missing. Installing CUDA Toolkit (this takes time)..."
        DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-cuda-toolkit
    fi
fi

# 4. COMPILATION
echo "⚡ Compiling CUDA Engine for Blackwell..."
# We use standard GGML_CUDA=on. llama.cpp is smart enough to detect B200/H100 arch.
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

echo "✅ B200 ENVIRONMENT READY."
echo "👉 SFTP your 'models' and 'dataset' folders now."