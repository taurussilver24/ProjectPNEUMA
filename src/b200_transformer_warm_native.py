"""
TRANSFORMER WARM MODE - Native PyTorch
Qwen 3B + Phi-4 14B
Load each model once, run all files through it
"""
import time
import json
import os
import re
import gc
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === CONFIG ===
PLATFORM = "NVIDIA B200 (Native PyTorch BF16)"
DTYPE = torch.bfloat16
FILE_LIMIT = 1000

# === MODELS ===
MODELS = [
    {
        "name": "Qwen2.5-3B",
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "trust_remote_code": False
    },
    {
        "name": "Phi-4-14B",
        "id": "microsoft/phi-4",
        "trust_remote_code": True
    }
]

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "dataset", "preprocessed.json")
LOG_DIR = os.path.join(BASE_DIR, "results")

# === CHAIN OF THOUGHT PROMPT ===
SYSTEM_PROMPT = """You are an expert research librarian. Analyze the document to extract metadata.

STEP 1: REASONING
Think about the document structure:
- Identify the main title
- Distinguish authors from affiliations
- Locate the DOI of THIS paper (not references)
- Synthesize the core contribution

STEP 2: EXTRACTION
Output the final metadata in this JSON format inside a code block:

```json
{
    "title": "Exact string",
    "authors": ["List of strings"],
    "doi": "String or null",
    "arxiv_id": "String or null",
    "keywords": ["List of strings"],
    "summary": "String"
}
```"""


def extract_json_from_cot(text):
    """Extract JSON from chain-of-thought response"""
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except:
        return {}


def run_transformer_warm():
    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(CACHE_PATH):
        print(f"❌ Cache not found: {CACHE_PATH}")
        return

    # Load texts
    print("📂 Loading pre-processed texts...")
    with open(CACHE_PATH) as f:
        texts_cache = json.load(f)
    filenames = sorted(list(texts_cache.keys()))[:FILE_LIMIT]
    print(f"✅ Loaded {len(filenames)} texts\n")

    # === MODEL LOOP ===
    for config in MODELS:
        name = config["name"]

        print("\n" + "=" * 70)
        print(f"🔥 {name.upper()} WARM MODE (CoT) | B200 NATIVE BENCHMARK")
        print(f"Platform: {PLATFORM}")
        print(f"Files: {len(filenames)}")
        print("=" * 70 + "\n")

        try:
            # Load model
            print(f"⚡ Loading {name} to B200...")
            start_load = time.time()

            tokenizer = AutoTokenizer.from_pretrained(
                config["id"],
                trust_remote_code=config["trust_remote_code"]
            )

            model = AutoModelForCausalLM.from_pretrained(
                config["id"],
                torch_dtype=DTYPE,
                device_map="auto",
                trust_remote_code=config["trust_remote_code"],
                attn_implementation="flash_attention_2"
            )

            load_time = time.time() - start_load
            vram_gb = torch.cuda.memory_allocated() / 1e9

            print(f"✅ Loaded in {load_time:.2f}s")
            print(f"📊 VRAM: {vram_gb:.2f}GB\n")

            # Inference loop
            model.eval()
            results = []

            for i, fname in enumerate(tqdm(filenames, desc=f"{name}")):
                text = texts_cache[fname]

                try:
                    prompt = f"{SYSTEM_PROMPT}\n\nDocument:\n{text}"

                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=3500
                    ).to("cuda")

                    start_infer = time.time()

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )

                    inference_time = time.time() - start_infer
                    generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
                    tps = generated_tokens / inference_time if inference_time > 0 else 0

                    # Decode and extract JSON
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    data = extract_json_from_cot(response)

                    results.append({
                        "filename": fname,
                        "title": data.get("title", "N/A"),
                        "authors": str(data.get("authors", "N/A")),
                        "doi": data.get("doi", "None"),
                        "arxiv_id": data.get("arxiv_id", "None"),
                        "keywords": str(data.get("keywords", "None")),
                        "summary": data.get("summary", "N/A"),
                        "tps": round(tps, 2),
                        "inference_time": round(inference_time, 3),
                        "tokens": generated_tokens,
                        "mode": "warm_cot",
                        "model": name,
                        "platform": PLATFORM
                    })

                    if (i + 1) % 50 == 0:
                        avg_tps = sum(r['tps'] for r in results[-50:]) / min(50, len(results))
                        print(f"\n[{i + 1}/{len(filenames)}] Avg TPS (last 50): {avg_tps:.2f}")

                except Exception as e:
                    print(f"\n❌ Error on {fname}: {e}")

            # Save results
            df = pd.DataFrame(results)
            output_csv = os.path.join(LOG_DIR, f"b200_{name.lower().replace('.', '_')}_warm_native.csv")
            df.to_csv(output_csv, index=False)

            successful = df[df['tps'] > 0]
            print(f"\n{'=' * 70}")
            print(f"📊 {name} WARM BENCHMARK COMPLETE")
            print(f"Saved: {output_csv}")
            if len(successful) > 0:
                print(f"Mean TPS: {successful['tps'].mean():.2f}")
                print(f"Median TPS: {successful['tps'].median():.2f}")
                print(f"Max TPS: {successful['tps'].max():.2f}")
            print(f"{'=' * 70}\n")

            # Cleanup
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            print("🧹 VRAM cleared\n")
            time.sleep(2)

        except Exception as e:
            print(f"❌ {name} FAILED: {e}\n")


if __name__ == "__main__":
    run_transformer_warm()