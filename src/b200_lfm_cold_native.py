"""
LFM-2.5 COLD MODE - Native PyTorch
Reload model for EACH file (measures load + inference time)
"""
import time
import json
import os
import gc
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# === CONFIG ===
PLATFORM = "NVIDIA B200 (Native PyTorch BF16)"
MODEL_ID = "LiquidAI/LFM-2.5-1.2B"
DTYPE = torch.bfloat16
FILE_LIMIT = 1000

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(BASE_DIR, "dataset", "preprocessed.json")
LOG_DIR = os.path.join(BASE_DIR, "results")

# === PROMPT ===
SYSTEM_PROMPT = """Extract paper metadata as JSON:
{
    "title": "Exact paper title",
    "authors": ["List of names"],
    "doi": "DOI string if found",
    "arxiv_id": "ID if found",
    "keywords": ["Technical tags"],
    "summary": "1-sentence abstract"
}
Output ONLY JSON."""

def run_lfm_cold():
    os.makedirs(LOG_DIR, exist_ok=True)

    if not os.path.exists(CACHE_PATH):
        print(f"❌ Cache not found: {CACHE_PATH}")
        return

    print("\n" + "=" * 70)
    print(f"❄️ LFM-2.5 COLD START | B200 NATIVE BENCHMARK")
    print(f"Platform: {PLATFORM}")
    print(f"Target: {FILE_LIMIT} Files (reload per file)")
    print("=" * 70 + "\n")

    # Load texts
    print("📂 Loading pre-processed texts...")
    with open(CACHE_PATH) as f:
        texts_cache = json.load(f)
    filenames = sorted(list(texts_cache.keys()))[:FILE_LIMIT]
    print(f"✅ Loaded {len(filenames)} texts\n")

    results = []

    print(f"🚀 Running cold start (reload per file)...\n")

    for i, fname in enumerate(tqdm(filenames, desc="Processing")):
        text = texts_cache[fname]

        try:
            # === COLD START: LOAD MODEL ===
            start_total = time.time()

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )

            model.eval()
            load_time = time.time() - start_total

            # === INFERENCE ===
            prompt = f"{SYSTEM_PROMPT}\n\nDocument:\n{text}\n\nJSON:"

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
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            inference_time = time.time() - start_infer
            total_time = time.time() - start_total
            generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tps = generated_tokens / inference_time if inference_time > 0 else 0

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(response[json_start:json_end])
                else:
                    data = {}
            except:
                data = {}

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
                "load_time": round(load_time, 3),
                "total_time": round(total_time, 3),
                "tokens": generated_tokens,
                "mode": "cold_start",
                "model": "LFM-2.5",
                "platform": PLATFORM
            })

            if (i + 1) % 10 == 0:
                avg_total = sum(r['total_time'] for r in results[-10:]) / min(10, len(results))
                print(f"\n[{i+1}/{len(filenames)}] Avg total time (last 10): {avg_total:.2f}s")

            # === CLEANUP (CRITICAL FOR COLD START) ===
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"\n❌ Error on {fname}: {e}")

    # === SAVE ===
    df = pd.DataFrame(results)
    output_csv = os.path.join(LOG_DIR, "b200_lfm_cold_native.csv")
    df.to_csv(output_csv, index=False)

    successful = df[df['tps'] > 0]
    print(f"\n{'=' * 70}")
    print(f"📊 LFM-2.5 COLD START COMPLETE")
    print(f"Saved: {output_csv}")
    if len(successful) > 0:
        print(f"Mean Load Time: {successful['load_time'].mean():.2f}s")
        print(f"Mean Inference Time: {successful['inference_time'].mean():.2f}s")
        print(f"Mean Total Time: {successful['total_time'].mean():.2f}s")
        print(f"Mean TPS: {successful['tps'].mean():.2f}")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    run_lfm_cold()