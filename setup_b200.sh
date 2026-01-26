#!/bin/bash

# Stop script on any error
set -e

echo "🔥 IGNITING B200 MANUAL SETUP..."

# 1. INSTALL DEPENDENCIES
echo "📦 Installing Python Libraries..."
python3 -m pip install --upgrade pip
# We need 'gdown' for Google Drive, plus your standard bench libs
pip install requests pandas pypdf huggingface_hub gdown

# 2. DATASET (Manual Drive Download)
echo "📚 Downloading Dataset from Google Drive..."

# File ID: 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc
# We use gdown CLI directly to handle the large file warning
gdown --id 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc -O dataset.zip

# 3. EXTRACT & FLATTEN
echo "📂 Processing Dataset..."
rm -rf dataset
mkdir -p dataset
mkdir -p temp_extract

if [ -f "dataset.zip" ]; then
    echo "   Unzipping..."
    unzip -q dataset.zip -d temp_extract

    echo "   Flattening files (Moving all PDFs to root dataset folder)..."
    # Find EVERY .pdf inside temp_extract and move to dataset/
    find temp_extract -name "*.pdf" -exec mv {} dataset/ \;

    # Cleanup
    rm -rf temp_extract dataset.zip

    COUNT=$(ls dataset/*.pdf 2>/dev/null | wc -l)
    echo "✅ Dataset ready: $COUNT PDFs in /dataset folder."
else
    echo "❌ CRITICAL: dataset.zip failed to download."
    exit 1
fi

# 4. DOWNLOAD MODELS (Using your src/download.py)
echo "🧠 Downloading Models..."
# We must CD into src so that '../models' resolves to the project root
cd src
python3 download.py
cd ..

# 5. ENGINE SETUP (Linux CUDA 12 Binary)
echo "⬇️ Downloading Llama.cpp (Linux CUDA 12)..."
rm -rf llama_dist
mkdir -p llama_dist

# Standard Ubuntu/CUDA12 build (b4650) from official release
wget -q --show-progress https://github.com/ggerganov/llama.cpp/releases/download/b4650/llama-b4650-bin-ubuntu-x64-cuda12.2.0.zip -O llama_dist/llama.zip

echo "   Unzipping Engine..."
unzip -q llama_dist/llama.zip -d llama_dist
chmod +x llama_dist/llama-server

echo "✅ SETUP COMPLETE."
echo "   Run './run_check.sh' to test the B200."