#!/bin/bash
# B200 DEPLOYMENT FROM YOUR GITHUB REPO
# Clones ProjectPNEUMA, replaces GGUF with native PyTorch, keeps CSV format

set -e

echo "🔥 B200 DEPLOYMENT FROM GITHUB"
echo "=============================="
echo ""

# 1. Clone repo
echo "📥 Cloning ProjectPNEUMA..."
rm -rf ProjectPNEUMA 2>/dev/null || true
git clone https://github.com/taurussilver24/ProjectPNEUMA.git
cd ProjectPNEUMA
git checkout H100Bench
echo "✅ Repo cloned"
echo ""

# 2. Clean old artifacts
echo "💥 Cleaning old GGUF artifacts..."
rm -rf models/* results/*
pkill -f python3 2>/dev/null || true
echo "✅ Clean"
echo ""

# 3. Install deps
echo "📦 Installing dependencies..."
pip install -q transformers accelerate pypdf pandas tqdm gdown
pip install flash-attn --no-build-isolation 2>/dev/null || echo "⚠️  Flash Attention skipped"
echo "✅ Dependencies ready"
echo ""

# 4. Download dataset (if not exists)
if [ ! -f "dataset/preprocessed.json" ]; then
    echo "⬇️ Downloading dataset..."
    if [ -z "$(ls -A dataset/*.pdf 2>/dev/null)" ]; then
        gdown --id 1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc -O dataset.zip
        unzip -q dataset.zip -d dataset/
        rm dataset.zip
    fi

    # Preprocess PDFs
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
print(f"Processing {len(files)} PDFs...")

with ThreadPoolExecutor(max_workers=32) as executor:
    texts = list(tqdm(executor.map(extract_text, files), total=len(files)))

cache = {os.path.basename(f): t for f, t in zip(files, texts)}

with open("dataset/preprocessed.json", "w") as f:
    json.dump(cache, f)

print(f"✅ Cached {len(cache)} texts")
PREPROCESS
else
    echo "✅ Dataset already preprocessed"
fi
echo ""

# 5. Download models
echo "📥 Downloading models (~10 min)..."
python3 << 'DOWNLOAD'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time

MODELS = [
    {"name": "LFM-2.5-1.2B", "id": "LiquidAI/LFM-2.5-1.2B", "trust": True},
    {"name": "Qwen2.5-3B", "id": "Qwen/Qwen2.5-3B-Instruct", "trust": False},
    {"name": "Phi-4-14B", "id": "microsoft/phi-4", "trust": True}
]

for i, config in enumerate(MODELS, 1):
    print(f"\n[{i}/3] {config['name']}...")
    start = time.time()

    tok = AutoTokenizer.from_pretrained(config["id"], trust_remote_code=config["trust"])
    mod = AutoModelForCausalLM.from_pretrained(
        config["id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=config["trust"],
        device_map="auto"
    )

    elapsed = time.time() - start
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  ✅ {elapsed:.1f}s, {vram:.2f}GB VRAM")

    del mod, tok
    torch.cuda.empty_cache()

print("\n✅ All models cached")
DOWNLOAD
echo ""

# 6. Create native PyTorch benchmarks (replaces old GGUF ones)
echo "📝 Creating native PyTorch benchmarks..."

# Backup old scripts
mkdir -p src/old_gguf_scripts
mv src/h100_*.py src/old_gguf_scripts/ 2>/dev/null || true

# Create new native scripts
cat > src/b200_lfm_warm.py << 'LFMWARM'
"""LFM WARM - Native PyTorch - Keeps original CSV format"""
import time, json, os, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PLATFORM = "NVIDIA B200 (Native PyTorch BF16)"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "dataset", "preprocessed.json")
LOG_DIR = os.path.join(BASE_DIR, "results")

print("\n" + "=" * 70)
print("🔥 LFM-2.5 WARM MODE | B200 BENCHMARK")
print("=" * 70 + "\n")

# Load texts
with open(CACHE_PATH) as f:
    texts_cache = json.load(f)
filenames = sorted(list(texts_cache.keys()))[:1000]
print(f"📂 Loaded {len(filenames)} texts\n")

# Load model
print("⚡ Loading LFM-2.5...")
tok = AutoTokenizer.from_pretrained("LiquidAI/LFM-2.5-1.2B", trust_remote_code=True)
mod = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM-2.5-1.2B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
mod.eval()
print("✅ Model loaded\n")

# Benchmark
results = []
for i, fname in enumerate(tqdm(filenames, desc="Processing")):
    text = texts_cache[fname]
    prompt = f"Extract metadata as JSON:\n{text}\nJSON:"

    try:
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3500).to("cuda")

        start = time.time()
        with torch.no_grad():
            outputs = mod.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tok.eos_token_id
            )

        inference_time = time.time() - start
        tokens = len(outputs[0]) - len(inputs["input_ids"][0])
        tps = tokens / inference_time if inference_time > 0 else 0

        # Decode and parse JSON
        response = tok.decode(outputs[0], skip_special_tokens=True)
        try:
            json_str = response[response.find("{"):response.rfind("}")+1]
            data = json.loads(json_str)
        except:
            data = {}

        # EXACT CSV FORMAT FROM YOUR REPO
        results.append({
            "filename": fname,
            "title": data.get("title", "N/A"),
            "authors": data.get("authors", "N/A"),
            "doi": data.get("doi", ""),
            "arxiv_id": data.get("arxiv_id", ""),
            "keywords": data.get("keywords", ""),
            "summary": data.get("summary", "N/A"),
            "tps": round(tps, 2),
            "model": "LFM-2.5",
            "platform": PLATFORM
        })

        if (i + 1) % 50 == 0:
            avg_tps = sum(r['tps'] for r in results[-50:]) / 50
            print(f"\n[{i+1}/{len(filenames)}] Avg TPS: {avg_tps:.2f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")

# Save
os.makedirs(LOG_DIR, exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(os.path.join(LOG_DIR, "b200_lfm_warm.csv"), index=False)

print(f"\n✅ Complete! Mean TPS: {df['tps'].mean():.2f}")
print(f"Saved to: results/b200_lfm_warm.csv\n")
LFMWARM

cat > src/b200_transformer_warm.py << 'TRANSWARM'
"""TRANSFORMER WARM - Native PyTorch - Keeps CSV format"""
import time, json, os, re, gc, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PLATFORM = "NVIDIA B200 (Native PyTorch BF16)"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "dataset", "preprocessed.json")
LOG_DIR = os.path.join(BASE_DIR, "results")

MODELS = [
    {"name": "Qwen2.5-3B", "id": "Qwen/Qwen2.5-3B-Instruct", "trust": False},
    {"name": "Phi-4-14B", "id": "microsoft/phi-4", "trust": True}
]

# Load texts
with open(CACHE_PATH) as f:
    texts_cache = json.load(f)
filenames = sorted(list(texts_cache.keys()))[:1000]

def extract_json(text):
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except: pass
    return {}

for config in MODELS:
    print("\n" + "=" * 70)
    print(f"🔥 {config['name'].upper()} WARM MODE | B200 BENCHMARK")
    print("=" * 70 + "\n")

    # Load model
    print(f"⚡ Loading {config['name']}...")
    tok = AutoTokenizer.from_pretrained(config["id"], trust_remote_code=config["trust"])
    mod = AutoModelForCausalLM.from_pretrained(
        config["id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=config["trust"],
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    mod.eval()
    print("✅ Model loaded\n")

    results = []
    for i, fname in enumerate(tqdm(filenames, desc=config['name'])):
        text = texts_cache[fname]
        prompt = f"Extract metadata as JSON with reasoning:\n{text}\n```json"

        try:
            inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3500).to("cuda")

            start = time.time()
            with torch.no_grad():
                outputs = mod.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id
                )

            inference_time = time.time() - start
            tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tps = tokens / inference_time if inference_time > 0 else 0

            response = tok.decode(outputs[0], skip_special_tokens=True)
            data = extract_json(response)

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A"),
                "authors": data.get("authors", "N/A"),
                "doi": data.get("doi", ""),
                "arxiv_id": data.get("arxiv_id", ""),
                "keywords": data.get("keywords", ""),
                "summary": data.get("summary", "N/A"),
                "tps": round(tps, 2),
                "model": config['name'],
                "platform": PLATFORM
            })

            if (i + 1) % 50 == 0:
                print(f"\n[{i+1}] Avg TPS: {sum(r['tps'] for r in results[-50:])/50:.2f}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    # Save
    df = pd.DataFrame(results)
    output = os.path.join(LOG_DIR, f"b200_{config['name'].lower().replace('.','_')}_warm.csv")
    df.to_csv(output, index=False)
    print(f"\n✅ {config['name']} done! Mean TPS: {df['tps'].mean():.2f}\n")

    del mod, tok
    torch.cuda.empty_cache()
    gc.collect()
TRANSWARM

echo "✅ Benchmark scripts created"
echo ""

# 7. Final instructions
echo "="
echo "✅ B200 READY TO BENCHMARK"
echo "="
echo ""
echo "📂 Repository: ProjectPNEUMA (H100Bench branch)"
echo "📂 Models cached: ~/.cache/huggingface/"
echo "📂 Dataset: dataset/preprocessed.json"
echo ""
echo "🚀 RUN BENCHMARKS:"
echo ""
echo "  # Warm mode (recommended first):"
echo "  python3 src/b200_lfm_warm.py"
echo "  python3 src/b200_transformer_warm.py"
echo ""
echo "  # Cold mode (optional, expensive):"
echo "  # (Create cold scripts if needed)"
echo ""
echo "Monitor GPU: watch -n 1 nvidia-smi"
echo ""
echo "CSV format: filename,title,authors,doi,arxiv_id,keywords,summary,tps,model,platform"
echo "Results: results/*.csv"
echo "="