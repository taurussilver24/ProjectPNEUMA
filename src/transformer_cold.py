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
FILE_LIMIT = 1000

# --- 2. MODEL QUEUE ---
MODEL_QUEUE = [
    {"name": "Qwen2.5-3B", "file": "qwen2.5-3b-instruct.Q4_K_M.gguf"},
    {"name": "Phi-4", "file": "phi-4.Q4_K_M.gguf"}
]

# --- 3. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "input")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# --- 4. SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are an expert research librarian. Analyze the document to extract metadata.

STEP 1: REASONING
First, think silently about the document structure:
- Identify the main title.
- Locate the DOI (must start with 10.).
- Distinguish the actual authors from affiliations.

STEP 2: EXTRACTION
Output the final metadata in this exact JSON format inside a code block:
{
    "title": "Exact string",
    "authors": ["List of strings"],
    "doi": "String or null",
    "arxiv_id": "String or null",
    "keywords": ["List of strings"],
    "summary": "One sentence summary"
}"""

def run_cold_benchmark(model_config):
    model_path = os.path.join(MODEL_DIR, model_config["file"])

    if not os.path.exists(model_path):
        print(f"❌ Error: {model_config['name']} not found at {model_path}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print("\n" + "=" * 70)
    print(f"🚀 FIRING {model_config['name']} 4090 COLD-START")
    print(f"Platform: {PLATFORM_NAME} | Batch: 4096")
    print("=" * 70 + "\n")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        llm = None

        try:
            # 1. READ PDF
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # 2. COLD START: Reloading weights for every single document
            start_time = time.time()
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,    # Full VRAM offload
                n_ctx=4096,
                n_batch=4096,       # High batch size for 4090 bandwidth
                n_ubatch=1024,
                flash_attn=True,    # Ada Lovelace supports this perfectly
                verbose=False
            )

            # 3. INFERENCE
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
            total_time = time.time() - start_time
            tps = tokens / total_time

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A"),
                "authors": ", ".join(data.get("authors", [])) if isinstance(data.get("authors"), list) else "N/A",
                "doi": data.get("doi", "None"),
                "tps": round(tps, 2),
                "total_time": round(total_time, 3),
                "tokens": tokens,
                "mode": "cold_start",
                "model": model_config["name"],
                "platform": PLATFORM_NAME
            })

            # Print status every 10 files
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f}")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

        finally:
            # 4. MEMORY PURGE
            # Critical for 4090 Cold Start to emulate "fresh boot" latency
            if llm:
                del llm
            gc.collect()

    # Save results
    output_csv = os.path.join(LOG_DIR, f"clash_{model_config['name']}_cold_RTX_4090.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Saved results to {output_csv}")

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    for model in MODEL_QUEUE:
        run_cold_benchmark(model)