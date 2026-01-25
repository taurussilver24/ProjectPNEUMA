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
FILE_LIMIT = 1000  # Full marathon execution

# --- 2. MULTI-MODEL QUEUE ---
MODEL_QUEUE = [
    {"name": "Qwen2.5-3B", "file": "qwen2.5-3b-instruct.Q4_K_M.gguf"},
    {"name": "Phi-4", "file": "phi-4.Q4_K_M.gguf"}
]

# --- 3. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "input")
LOG_DIR = os.path.join(BASE_DIR, "logs")

SYSTEM_PROMPT = """You are an expert research librarian. Output metadata in strict JSON.
{
    "title": "Exact string",
    "authors": ["List of strings"],
    "doi": "String or null",
    "arxiv_id": "String or null",
    "keywords": ["List of strings"],
    "summary": "One sentence summary"
}"""


def run_warm_benchmark(model_config):
    model_path = os.path.join(MODEL_DIR, model_config["file"])
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_config['name']}: Not found at {model_path}")
        return

    print(f"\n🚀 ACTIVATING 4090 WARM MODE: {model_config['name']}")

    # LOAD MODEL ONCE (Persistent Session)
    # n_batch=4096 is optimal for Ada Lovelace's memory bandwidth
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=4096,
        n_ubatch=1024,
        flash_attn=True,
        verbose=False
    )

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            text = PdfReader(path).pages[0].extract_text()[:3500]
            start_time = time.time()

            resp = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            data = json.loads(resp['choices'][0]['message']['content'])
            tokens = resp['usage']['completion_tokens']
            inference_time = time.time() - start_time
            tps = tokens / inference_time

            # Clear KV cache to maintain context purity
            llm.reset()

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A"),
                "authors": ", ".join(data.get("authors", [])) if isinstance(data.get("authors"), list) else "N/A",
                "doi": data.get("doi", "None"),
                "tps": round(tps, 2),
                "inference_time": round(inference_time, 3),
                "model": model_config["name"],
                "platform": PLATFORM_NAME,
                "mode": "warm"
            })

            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i + 1}/{len(files)}] {fname} | Speed: {tps:.2f} TPS")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

    # Cleanup VRAM before next model in queue
    del llm
    gc.collect()

    output_csv = os.path.join(LOG_DIR, f"clash_{model_config['name']}_warm_RTX_4090.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Marathon Complete. Results: {output_csv}")


if __name__ == "__main__":
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    for m in MODEL_QUEUE:
        run_warm_benchmark(m)