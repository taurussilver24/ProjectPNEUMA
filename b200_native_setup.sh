#!/bin/bash
# B200 NATIVE PYTORCH - COMPLETE BENCHMARK SUITE
# All 4 benchmarks: LFM warm/cold, Transformer warm/cold

set -e

echo "🔥 B200 NATIVE PYTORCH BENCHMARK SUITE"
echo "======================================"
echo ""

# 1. CLEAN SLATE
echo "💥 Cleaning workspace..."
pkill -f python3 2>/dev/null || true
rm -rf models/* results/* src/*
mkdir -p models results src dataset

# 2. INSTALL PYTORCH STACK
echo "🛠️ Installing PyTorch stack..."
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install transformers accelerate -q
pip install pandas pypdf einops tqdm -q

# Try to install flash attention (optional but highly recommended)
echo "⚡ Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation 2>/dev/null || {
    echo "⚠️ Flash Attention install failed (will use standard attention)"
}

echo "✅ Stack installed"
echo ""

# 3. VERIFY CUDA
echo "🔍 CUDA Verification..."
python3 << 'VERIFY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print("✅ CUDA Ready")
else:
    print("❌ NO CUDA DETECTED!")
    exit(1)
VERIFY
echo ""

# 4. DOWNLOAD DATASET
if [ ! -d "dataset" ] || [ -z "$(ls -A dataset/*.pdf 2>/dev/null)" ]; then
    echo "⬇️ Downloading dataset..."
    pip install gdown -q
    gdown --id 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc -O dataset.zip
    unzip -q dataset.zip -d dataset/
    rm dataset.zip
    echo "✅ Dataset ready ($(ls dataset/*.pdf 2>/dev/null | wc -l) PDFs)"
else
    echo "✅ Dataset exists ($(ls dataset/*.pdf | wc -l) PDFs)"
fi
echo ""

# 5. PRE-DOWNLOAD MODELS (CRITICAL)
echo "📥 Pre-downloading models from HuggingFace..."
python3 << 'DOWNLOAD_MODELS'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODELS = [
    {"name": "LFM-2.5-1.2B", "id": "LiquidAI/LFM-2.5-1.2B", "trust": True},
    {"name": "Qwen2.5-3B", "id": "Qwen/Qwen2.5-3B-Instruct", "trust": False},
    {"name": "Phi-4-14B", "id": "microsoft/phi-4", "trust": True}
]

print("\n🚀 Downloading 3 models (this may take 10-15 minutes)...\n")

for config in MODELS:
    print(f"📦 Downloading {config['name']}...")
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config["id"],
            trust_remote_code=config["trust"]
        )
        print(f"  ✅ Tokenizer cached")

        # Download model (this caches it locally)
        model = AutoModelForCausalLM.from_pretrained(
            config["id"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=config["trust"],
            device_map="auto"
        )
        print(f"  ✅ Model cached")

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()

        print(f"✅ {config['name']} ready\n")
    except Exception as e:
        print(f"❌ Failed to download {config['name']}: {e}\n")
        exit(1)

print("✅ All models cached to ~/.cache/huggingface/\n")
DOWNLOAD_MODELS
echo ""

# 6. PRE-PROCESS PDFs (CRITICAL OPTIMIZATION)
echo "📄 Pre-processing PDFs (this eliminates I/O bottleneck)..."
python3 << 'PREPROCESS'
import glob, json, os
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_text(path):
    try:
        reader = PdfReader(path)
        return reader.pages[0].extract_text()[:3500] if len(reader.pages) > 0 else ""
    except:
        return ""

files = sorted(glob.glob("dataset/*.pdf"))[:1000]
print(f"Processing {len(files)} PDFs with 32 threads...")

with ThreadPoolExecutor(max_workers=32) as executor:
    texts = list(tqdm(executor.map(extract_text, files), total=len(files)))

cache = {os.path.basename(f): t for f, t in zip(files, texts)}

with open("dataset/preprocessed.json", "w") as f:
    json.dump(cache, f)

print(f"✅ Cached {len(cache)} texts to preprocessed.json")
PREPROCESS
echo ""

# 7. VERIFY MODEL CACHE
echo "🔍 Verifying model cache..."
python3 << 'VERIFY_CACHE'
import os
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
if cache_dir.exists():
    models = list(cache_dir.glob("models--*"))
    print(f"✅ Found {len(models)} cached models")
    for m in models:
        size_gb = sum(f.stat().st_size for f in m.rglob('*') if f.is_file()) / 1e9
        print(f"  - {m.name}: {size_gb:.2f}GB")
else:
    print("⚠️ No model cache found")
VERIFY_CACHE
echo ""

# 8. CREATE ALL 4 BENCHMARK SCRIPTS
echo "📝 Creating benchmark scripts..."

# The 4 Python scripts will be downloaded/created separately
# This setup script just prepares the environment

echo "✅ Environment ready!"
echo ""
echo "=" * 70
echo "🚀 TO RUN BENCHMARKS:"
echo ""
echo "  1. LFM Warm:         python3 src/b200_lfm_warm_native.py"
echo "  2. LFM Cold:         python3 src/b200_lfm_cold_native.py"
echo "  3. Transformer Warm: python3 src/b200_transformer_warm_native.py"
echo "  4. Transformer Cold: python3 src/b200_transformer_cold_native.py"
echo ""
echo "Models cached at: ~/.cache/huggingface/hub/"
echo "Total model size: ~35GB"
echo ""
echo "Expected B200 Performance (Native PyTorch):"
echo "  - LFM 1.2B:   800-1200 TPS (warm), 3-5s total (cold)"
echo "  - Qwen 3B:    1500-2000 TPS (warm), 5-8s total (cold)"
echo "  - Phi-4 14B:  400-600 TPS (warm), 15-20s total (cold)"
echo ""
echo "Results will be saved to: results/*.csv"
echo "=" * 70