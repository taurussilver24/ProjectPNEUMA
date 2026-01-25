import time
import json
import os
import glob
import gc
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- 1. RESEARCH CONFIGURATION ---
PLATFORM_NAME = "NVIDIA RTX 4090 (Ada Lovelace)"
MODEL_FILE = "lfm-2.5-1.2b.Q4_K_M.gguf"
FILE_LIMIT = 1000 # Set for full marathon

# --- 2. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_FILE)
DATA_DIR = os.path.join(BASE_DIR, "data", "input")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# --- 3. THE "LIQUID-OPTIMIZED" PROMPT ---
# This prompt uses the Reasoning + Extraction structure which LFM excels at.
SYSTEM_PROMPT = """You are an expert research librarian. Analyze the document to extract metadata.

STEP 1: REASONING
First, think silently about the document structure:
- Identify the main title.
- Locate the DOI (must start with 10.).
- Synthesize the core contribution.

STEP 2: EXTRACTION
Output the metadata in strict JSON format:
{
    "title": "Exact string",
    "authors": ["List of strings"],
    "doi": "String or null",
    "arxiv_id": "String or null",
    "keywords": ["List of strings"],
    "summary": "One sentence summary"
}"""

def run_lfm_warm():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    print("\n" + "=" * 70)
    print(f"🔥 RTX 4090 LFM-2.5 REDEMPTION: WARM MODE")
    print(f"Platform: {PLATFORM_NAME} | Models: {MODEL_FILE}")
    print("=" * 70 + "\n")

    # 4. LOAD MODEL ONCE (Persistent Session)
    print("Engaging Ada Lovelace Cores...")
    start_load = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,    # Full 4090 offload
        n_ctx=4096,         # standard context window
        n_batch=4096,       # Maximize GDDR6X bandwidth
        n_ubatch=1024,      # Optimized for Ada architecture
        flash_attn=True,    # Essential for efficiency
        verbose=False
    )
    load_time = time.time() - start_load
    print(f"✅ 4090 Warm Session Active ({load_time:.2f}s load time)\n")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            # 5. READ PDF
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # 6. INFERENCE (Persistent Model)
            start_time = time.time()
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            # Performance Metrics
            data = json.loads(resp['choices'][0]['message']['content'])
            tokens = resp['usage']['completion_tokens']
            inference_time = time.time() - start_time
            tps = tokens / inference_time

            # 7. KV-CACHE FLUSH
            llm.reset()

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A"),
                "authors": ", ".join(data.get("authors", [])) if isinstance(data.get("authors"), list) else "N/A",
                "doi": data.get("doi", "None"),
                "arxiv_id": data.get("arxiv_id", "None"),
                "tps": round(tps, 2),
                "inference_time": round(inference_time, 3),
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

            print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f} (Warm)")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

    # Save results
    output_csv = os.path.join(LOG_DIR, f"clash_LFM_warm_RTX_4090.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n✅ Results saved: {output_csv}")

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    run_lfm_warm()