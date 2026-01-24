import time
import json
import os
import glob
import gc
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- CONFIGURATION ---
PLATFORM_NAME = "NVIDIA H100 SXM (CUDA)"
MODEL_FILE = "lfm.gguf"  # Matches setup_h100.sh filename
FILE_LIMIT = 1000  # Full dataset

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_FILE)
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "results")

# --- SYSTEM PROMPT: BASELINE (PROMPT 1) ---
# CRITICAL: We strictly use Prompt 1 for LFM.
# Prompt 3 (CoT) causes "Instruction Collapse" on this model architecture.
SYSTEM_PROMPT = """You are a metadata extraction system. Extract fields into JSON:
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
    # 1. Validation
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print("\n" + "=" * 70)
    print(f"❄️ LFM-2.5 (Liquid) COLD START | H100 BENCHMARK")
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Target:   {len(files)} Files")
    print(f"Prompt:   Baseline (Strict JSON)")
    print("=" * 70 + "\n")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        llm = None

        try:
            # 2. Read PDF
            reader = PdfReader(path)
            if len(reader.pages) > 0:
                text = reader.pages[0].extract_text()[:3500]
            else:
                text = ""

            # 3. COLD START INIT (The "Loading" Penalty)
            start_total = time.time()

            # Re-initialize the engine from scratch for every file
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,  # Full Offload
                main_gpu=0,
                n_ctx=4096,
                n_batch=512,
                verbose=False
            )

            # 4. Inference
            start_infer = time.time()
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            # Metrics Calculation
            end_time = time.time()
            inference_time = end_time - start_infer  # Pure generation time
            total_time = end_time - start_total  # Load + Gen time

            content = resp['choices'][0]['message']['content']
            tokens = resp['usage']['completion_tokens']

            # TPS for Cold Start is usually calculated on Total Time (Load + Gen)
            # But to compare fairly with Warm, we often record both.
            # Here we record Pure Inference TPS to see if "State" affects speed,
            # but note the Total Time for the "Latency" argument.
            tps_infer = tokens / inference_time if inference_time > 0 else 0

            # Parse Data
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {}

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A"),
                "authors": str(data.get("authors", "N/A")),
                "doi": data.get("doi", "None"),
                "arxiv_id": data.get("arxiv_id", "None"),
                "keywords": str(data.get("keywords", "None")),
                "summary": data.get("summary", "N/A"),
                "tps": round(tps_infer, 2),
                "inference_time": round(inference_time, 3),
                "total_time": round(total_time, 3),  # Unique to Cold Start
                "tokens": tokens,
                "mode": "cold_start",
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

            # Progress output
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps_infer:.2f} | Load+Gen: {total_time:.2f}s")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

        finally:
            # 5. MEMORY PURGE (The "Destruction")
            if llm:
                del llm
            gc.collect()
            # On CUDA, this frees the pointer, though PyTorch/CUDA sometimes
            # holds a cache. llama-cpp-python usually releases well.

    # 6. Save Results
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    output_csv = os.path.join(LOG_DIR, "h100_lfm_baseline_cold.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("❄️ COLD START BENCHMARK COMPLETE")
    print(f"Saved to: {output_csv}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_lfm_cold()