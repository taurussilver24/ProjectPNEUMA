import time
import json
import os
import glob
import gc
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- 1. RESEARCH CONFIGURATION ---
PLATFORM_NAME = "NVIDIA RTX 5090 (CUDA 12.8)"
MODEL_FILE = "lfm-2.5-1.2b.Q4_K_M.gguf"
FILE_LIMIT = 1000  # Set for the full marathon

# --- 2. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Adjusted for the RunPod setup.sh path
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_FILE)
DATA_DIR = os.path.join(BASE_DIR, "data", "input")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Optimized Prompt: Strict JSON for 5090 throughput
SYSTEM_PROMPT = """You are an expert research librarian. Output metadata in strict JSON.
{
    "title": "Exact string",
    "authors": ["List of strings"],
    "doi": "String or null",
    "arxiv_id": "String or null",
    "keywords": ["List of strings"],
    "summary": "One sentence"
}"""

def run_lfm_marathon():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: LFM model not found at {MODEL_PATH}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print(f"\n--- 🚀 FIRING RTX 5090 COLD-START MARATHON ({len(files)} files) ---")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        llm = None

        try:
            # 1. READ PDF (Limit to first 3500 chars for speed)
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # 2. COLD START: Optimized for Blackwell (sm_120)
            start_time = time.time()
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,    # Force 5090 offload
                n_ctx=4096,         # standard context
                n_batch=2048,       # High batch for GDDR7
                n_ubatch=512,       # Micro-batch for core stability
                flash_attn=True,    # Tensor core engagement
                verbose=False
            )

            # 3. INFERENCE
            resp = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            # Calculate Performance
            data = json.loads(resp['choices'][0]['message']['content'])
            tokens = resp['usage']['completion_tokens']
            inference_duration = time.time() - start_time
            tps = tokens / inference_duration

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A"),
                "authors": ", ".join(data.get("authors", [])) if isinstance(data.get("authors"), list) else "N/A",
                "doi": data.get("doi", "None"),
                "arxiv_id": data.get("arxiv_id", "None"),
                "tps": round(tps, 2),
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

            print(f"[{i + 1}/{len(files)}] {fname} | Speed: {tps:.2f} TPS | {tokens} tokens")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

        finally:
            # 4. HARD MEMORY PURGE: Necessary for 5090 stability in Cold Start
            if llm:
                del llm
            gc.collect()
            # On CUDA 12.8, this helps prevent the 'context leak' seen on earlier drivers

    output_csv = os.path.join(LOG_DIR, f"clash_LFM_Cold_{PLATFORM_NAME.replace(' ', '_')}.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n✅ LFM Cold Marathon Complete. Data saved: {output_csv}")

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    run_lfm_marathon()