"""
LFM-2.5 WARM MODE - Native PyTorch
Load model once, run all 1000 files through it
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


def run_lfm_warm():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Validation
    if not os.path.exists(CACHE_PATH):
        print(f"❌ Cache not found: {CACHE_PATH}")
        print("   Run setup script first!")
        return

    print("\n" + "=" * 70)
    print(f"🔥 LFM-2.5 WARM MODE | B200 NATIVE BENCHMARK")
    print(f"Platform: {PLATFORM}")
    print(f"Target: {FILE_LIMIT} Files")
    print("=" * 70 + "\n")

    # Load pre-processed texts
    print("📂 Loading pre-processed texts...")
    with open(CACHE_PATH) as f:
        texts_cache = json.load(f)
    filenames = sorted(list(texts_cache.keys()))[:FILE_LIMIT]
    print(f"✅ Loaded {len(filenames)} texts\n")

    # === LOAD MODEL (WARM START) ===
    print(f"⚡ Loading LFM-2.5 to B200...")
    start_load = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    load_time = time.time() - start_load
    vram_gb = torch.cuda.memory_allocated() / 1e9

    print(f"✅ Loaded in {load_time:.2f}s")
    print(f"📊 VRAM: {vram_gb:.2f}GB\n")

    # === INFERENCE LOOP ===
    model.eval()
    results = []

    print(f"🚀 Running warm inference on {len(filenames)} files...\n")

    for i, fname in enumerate(tqdm(filenames, desc="Processing")):
        text = texts_cache[fname]

        try:
            # Build prompt
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
            generated_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
            tps = generated_tokens / inference_time if inference_time > 0 else 0

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Try to parse JSON from response
            try:
                # Extract JSON if embedded in text
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
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
                "tokens": generated_tokens,
                "mode": "warm",
                "model": "LFM-2.5",
                "platform": PLATFORM
            })

            if (i + 1) % 50 == 0:
                avg_tps = sum(r['tps'] for r in results[-50:]) / min(50, len(results))
                print(f"\n[{i + 1}/{len(filenames)}] Avg TPS (last 50): {avg_tps:.2f}")

        except Exception as e:
            print(f"\n❌ Error on {fname}: {e}")

    # === SAVE RESULTS ===
    df = pd.DataFrame(results)
    output_csv = os.path.join(LOG_DIR, "b200_lfm_warm_native.csv")
    df.to_csv(output_csv, index=False)

    # Summary
    successful = df[df['tps'] > 0]
    print(f"\n{'=' * 70}")
    print(f"📊 LFM-2.5 WARM BENCHMARK COMPLETE")
    print(f"Saved: {output_csv}")
    if len(successful) > 0:
        print(f"Mean TPS: {successful['tps'].mean():.2f}")
        print(f"Median TPS: {successful['tps'].median():.2f}")
        print(f"Max TPS: {successful['tps'].max():.2f}")
        print(f"Min TPS: {successful['tps'].min():.2f}")
    print(f"{'=' * 70}\n")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    run_lfm_warm()