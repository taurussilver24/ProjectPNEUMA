#!/bin/bash

# Stop on error
set -e

echo "🔥 IGNITING B200 MANUAL SETUP (Ubuntu 24.04 Hardened)..."

# 0. UPDATE SYSTEM TOOLS (Critical for RunPod/Ubuntu 24.04)
echo "🛠️  Updating System Packages..."
apt-get update -y
apt-get install -y unzip wget git

# 1. INSTALL DEPENDENCIES (Bypassing PEP 668)
echo "📦 Installing Python Libraries..."
python3 -m pip install --upgrade pip --break-system-packages
# Added --break-system-packages to bypass Ubuntu 24.04 restriction
pip install requests pandas pypdf huggingface_hub gdown --break-system-packages

# 2. DATASET (From Google Drive)
echo "📚 Downloading Dataset from Google Drive..."
gdown --id 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc -O dataset.zip

# 3. EXTRACT & FLATTEN
echo "📂 Processing Dataset..."
rm -rf dataset
mkdir -p dataset
mkdir -p temp_extract

if [ -f "dataset.zip" ]; then
    echo "   Unzipping..."
    unzip -q dataset.zip -d temp_extract

    echo "   Flattening files..."
    find temp_extract -name "*.pdf" -exec mv {} dataset/ \;

    # Cleanup
    rm -rf temp_extract dataset.zip

    COUNT=$(ls dataset/*.pdf 2>/dev/null | wc -l)
    echo "✅ Dataset ready: $COUNT PDFs in /dataset folder."
else
    echo "❌ CRITICAL: dataset.zip failed to download."
    exit 1
fi

# 4. DOWNLOAD MODELS
echo "🧠 Downloading Models..."
cd src
python3 download.py
cd ..

# 5. ENGINE SETUP (Linux CUDA 12 Binary)
echo "⬇️ Downloading Llama.cpp (Linux CUDA 12)..."
rm -rf llama_dist
mkdir -p llama_dist

# Using the CUDA 12 build (Compatible with CUDA 12.8)
wget -q --show-progress https://github.com/ggerganov/llama.cpp/releases/download/b4650/llama-b4650-bin-ubuntu-x64-cuda12.2.0.zip -O llama_dist/llama.zip

echo "   Unzipping Engine..."
unzip -q llama_dist/llama.zip -d llama_dist
chmod +x llama_dist/llama-server

echo "✅ SETUP COMPLETE."
echo "   Run './run_check.sh' to verify B200 detection."