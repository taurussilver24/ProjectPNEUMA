#!/bin/bash
# B200 NATIVE PYTORCH - OPTIMIZED FOR RUNPOD IMAGE
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# PyTorch 2.8.0 + CUDA 12.8.1 already installed

set -e

echo "🔥 B200 NATIVE PYTORCH BENCHMARK SUITE"
echo "RunPod Image: pytorch:1.0.2-cu1281-torch280-ubuntu2404"
echo "======================================================"
echo ""

# 1. CLEAN WORKSPACE
echo "💥 Cleaning workspace..."
pkill -f python3 2>/dev/null || true
rm -rf models results src 2>/dev/null || true
mkdir -p models results src dataset

# 2. VERIFY PYTORCH + CUDA (already installed in image)
echo "🔍 Verifying PyTorch + CUDA..."
python3 << 'VERIFY'
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ VRAM: {vram_gb:.1f}GB")
    if "B200" in torch.cuda.get_device_name(0) or vram_gb > 150:
        print(f"✅ B200 DETECTED!")
else:
    print("❌ NO CUDA!")
    exit(1)
VERIFY
echo ""

# 3. INSTALL ONLY MISSING PACKAGES
echo "📦 Installing additional packages..."
pip install -q transformers accelerate pypdf tqdm gdown 2>/dev/null || {
    pip install --upgrade pip
    pip install transformers accelerate pypdf tqdm gdown
}

# Try flash attention (optional but recommended)
echo "⚡ Attempting Flash Attention 2 install..."
pip install flash-attn --no-build-isolation 2>/dev/null && echo "✅ Flash Attention 2 installed" || echo "⚠️  Flash Attention skipped (optional)"
echo ""

# 4. DOWNLOAD DATASET
if [ ! -f "dataset/preprocessed.json" ]; then
    if [ -z "$(ls -A dataset/*.pdf 2>/dev/null)" ]; then
        echo "⬇️ Downloading dataset..."
        gdown --id 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc -O dataset.zip
        unzip -q dataset.zip -d dataset/
        rm dataset.zip
        echo "✅ Dataset extracted ($(ls dataset/*.pdf 2>/dev/null | wc -l) PDFs)"
    else
        echo "✅ Dataset exists ($(ls dataset/*.pdf | wc -l) PDFs)"
    fi

    # 5. PRE-PROCESS PDFs
    echo "📄 Pre-processing PDFs..."
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

print(f"✅ Cached {len(cache)} texts")
PREPROCESS
else
    echo "✅ Preprocessed cache exists"
fi
echo ""

# 6. PRE-DOWNLOAD MODELS
echo "📥 Pre-downloading models from HuggingFace..."
echo "This will take 8-12 minutes for ~37GB total"
echo ""

python3 << 'DOWNLOAD_MODELS'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

MODELS = [
    {"name": "LFM-2.5-1.2B", "id": "LiquidAI/LFM-2.5-1.2B", "trust": True},
    {"name": "Qwen2.5-3B", "id": "Qwen/Qwen2.5-3B-Instruct", "trust": False},
    {"name": "Phi-4-14B", "id": "microsoft/phi-4", "trust": True}
]

for i, config in enumerate(MODELS, 1):
    print(f"\n[{i}/3] Downloading {config['name']}...")
    print("-" * 60)

    try:
        start = time.time()

        # Download tokenizer
        print("  📥 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config["id"],
            trust_remote_code=config["trust"]
        )
        print("  ✅ Tokenizer cached")

        # Download model
        print("  📥 Model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            config["id"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=config["trust"],
            device_map="auto"
        )

        elapsed = time.time() - start
        vram_gb = torch.cuda.memory_allocated() / 1e9

        print(f"  ✅ Model loaded ({elapsed:.1f}s)")
        print(f"  📊 VRAM: {vram_gb:.2f}GB")

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()

        print(f"✅ {config['name']} ready")

    except Exception as e:
        print(f"❌ Failed: {e}")
        exit(1)

print("\n✅ All models cached to ~/.cache/huggingface/")
DOWNLOAD_MODELS
echo ""

# 7. VERIFY CACHE
echo "🔍 Verifying model cache..."
python3 << 'VERIFY_CACHE'
from pathlib import Path

cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
if cache_dir.exists():
    models = list(cache_dir.glob("models--*"))
    total_size = 0
    print(f"✅ Found {len(models)} cached models:")
    for m in models:
        size_gb = sum(f.stat().st_size for f in m.rglob('*') if f.is_file()) / 1e9
        total_size += size_gb
        model_name = m.name.replace("models--", "").replace("--", "/")
        print(f"  - {model_name}: {size_gb:.2f}GB")
    print(f"\n📊 Total cache: {total_size:.2f}GB")
else:
    print("⚠️ No model cache found")
VERIFY_CACHE
echo ""

# 8. FINAL STATUS
echo "="
echo "✅ B200 ENVIRONMENT READY"
echo "="
echo ""
echo "📂 Structure:"
echo "  - Models: ~/.cache/huggingface/hub/ (~37GB)"
echo "  - Dataset: dataset/preprocessed.json (1000 PDFs)"
echo "  - Scripts: src/*.py"
echo "  - Results: results/*.csv"
echo ""
echo "🚀 TO RUN BENCHMARKS:"
echo ""
echo "  # Quick test (recommended first):"
echo "  python3 src/b200_lfm_warm_native.py"
echo ""
echo "  # Full suite:"
echo "  python3 src/b200_lfm_warm_native.py"
echo "  python3 src/b200_lfm_cold_native.py"
echo "  python3 src/b200_transformer_warm_native.py"
echo "  python3 src/b200_transformer_cold_native.py"
echo ""
echo "Expected Performance:"
echo "  - LFM 1.2B:   800-1200 TPS (warm)"
echo "  - Qwen 3B:    1500-2000 TPS (warm)"
echo "  - Phi-4 14B:  400-600 TPS (warm)"
echo ""
echo "Monitor GPU: watch -n 1 nvidia-smi"
echo "="