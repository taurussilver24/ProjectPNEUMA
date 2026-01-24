#!/bin/bash
# AMD TEST - Model Download Verification Only
# Tests if models can be downloaded from HuggingFace
# No GPU inference, just download + tokenizer test

set -e

echo "🧪 AMD TEST - Model Download Verification"
echo "=========================================="
echo ""

# 1. Install minimal requirements
echo "📦 Installing dependencies..."
pip install -q transformers torch tqdm 2>/dev/null || {
    pip install --upgrade pip
    pip install transformers torch tqdm
}
echo "✅ Dependencies ready"
echo ""

# 2. Test model downloads
python3 << 'TEST_DOWNLOADS'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import time

MODELS = [
    {
        "name": "LFM-2.5-1.2B",
        "id": "LiquidAI/LFM-2.5-1.2B",
        "trust": True,
        "expected_size_gb": 2.5
    },
    {
        "name": "Qwen2.5-3B",
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "trust": False,
        "expected_size_gb": 6.5
    },
    {
        "name": "Phi-4-14B",
        "id": "microsoft/phi-4",
        "trust": True,
        "expected_size_gb": 28
    }
]

print("\n🚀 TESTING MODEL DOWNLOADS")
print("=" * 60)
print("Note: This will download ~37GB total")
print("First run will be slow, subsequent runs use cache")
print("=" * 60)
print("")

results = []

for i, config in enumerate(MODELS, 1):
    print(f"\n[{i}/3] Testing {config['name']}...")
    print("-" * 60)

    try:
        # Test tokenizer download
        print("  📥 Downloading tokenizer...")
        start = time.time()

        tokenizer = AutoTokenizer.from_pretrained(
            config["id"],
            trust_remote_code=config["trust"]
        )

        tokenizer_time = time.time() - start
        print(f"  ✅ Tokenizer ready ({tokenizer_time:.1f}s)")

        # Test model download
        print(f"  📥 Downloading model (~{config['expected_size_gb']:.1f}GB)...")
        start = time.time()

        model = AutoModelForCausalLM.from_pretrained(
            config["id"],
            torch_dtype=torch.float16,  # Use FP16 for AMD compatibility
            trust_remote_code=config["trust"],
            device_map="cpu",  # Force CPU for AMD test
            low_cpu_mem_usage=True
        )

        download_time = time.time() - start
        print(f"  ✅ Model downloaded ({download_time:.1f}s)")

        # Quick tokenizer test
        print("  🧪 Testing tokenizer...")
        test_text = "Hello, this is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        decoded = tokenizer.decode(tokens["input_ids"][0])
        print(f"  ✅ Tokenizer works (encoded {len(tokens['input_ids'][0])} tokens)")

        # Check cache size
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dirs = list(cache_dir.glob(f"models--{config['id'].replace('/', '--')}*"))

        if model_dirs:
            size_bytes = sum(f.stat().st_size for f in model_dirs[0].rglob('*') if f.is_file())
            size_gb = size_bytes / 1e9
            print(f"  📊 Cached size: {size_gb:.2f}GB")

        # Clean up to save RAM
        del model, tokenizer

        results.append({
            "name": config['name'],
            "status": "✅ SUCCESS",
            "download_time": f"{download_time:.1f}s",
            "size": f"{size_gb:.2f}GB" if model_dirs else "Unknown"
        })

        print(f"  ✅ {config['name']} TEST PASSED")

    except Exception as e:
        error_msg = str(e)[:100]
        results.append({
            "name": config['name'],
            "status": "❌ FAILED",
            "error": error_msg
        })
        print(f"  ❌ FAILED: {error_msg}")

# Summary
print("\n" + "=" * 60)
print("📊 DOWNLOAD TEST SUMMARY")
print("=" * 60)

for r in results:
    if r["status"] == "✅ SUCCESS":
        print(f"\n{r['status']} {r['name']}")
        print(f"   Download time: {r['download_time']}")
        print(f"   Cached size: {r['size']}")
    else:
        print(f"\n{r['status']} {r['name']}")
        print(f"   Error: {r.get('error', 'Unknown')}")

# Final verdict
success_count = sum(1 for r in results if "SUCCESS" in r["status"])
print("\n" + "=" * 60)

if success_count == len(MODELS):
    print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY")
    print("\nYou're ready to deploy to B200!")
    print("Models are cached at: ~/.cache/huggingface/hub/")
    print("\nNext steps:")
    print("  1. Copy these cached models to B200 (optional, saves time)")
    print("  2. Or let B200 re-download (models will cache there)")
else:
    print(f"⚠️  {success_count}/{len(MODELS)} models downloaded")
    print("\nCheck your internet connection and HuggingFace access")
    print("Some models may require authentication:")
    print("  huggingface-cli login")

print("=" * 60)
TEST_DOWNLOADS

echo ""
echo "🏁 Test complete!"
echo ""
echo "Cache location: ~/.cache/huggingface/hub/"
echo "Total download: ~37GB"