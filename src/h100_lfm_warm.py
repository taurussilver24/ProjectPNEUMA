import time
import json
import os
import glob
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- CONFIGURATION ---
PLATFORM_NAME = "NVIDIA H100 SXM (CUDA)"
MODEL_FILE = "lfm.gguf"  # Matches the 'wget' filename in setup_h100.sh
FILE_LIMIT = 1000  # Run the full dataset

# --- PATHS ---
# Uses relative paths for the Cloud Environment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project Root
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_FILE)
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "results")  # Standardized to 'results' folder

# --- SYSTEM PROMPT: BASELINE (PROMPT 1) ---
# CRITICAL: We use Prompt 1 for LFM because CoT (Prompt 3) causes
# "Instruction Collapse" (hallucinating placeholders) on this specific model.
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


def run_lfm_warm():
    # 1. Validation
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print(f"   Did you run 'bash setup_h100.sh'?")
        return

    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset not found at {DATA_DIR}")
        return

    print("\n" + "=" * 70)
    print(f"🔥 LFM-2.5 (Liquid) WARM MODE | H100 BENCHMARK")
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Target:   {FILE_LIMIT} Files")
    print(f"Prompt:   Baseline (Strict JSON)")
    print("=" * 70 + "\n")

    # 2. LOAD MODEL ONCE (WARM MODE)
    print("⚡ Loading LFM-2.5 to VRAM...")
    start_load = time.time()

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Offload ALL layers to H100
        main_gpu=0,
        n_ctx=4096,  # LFM context window
        n_batch=512,  # High batch size for CUDA speed
        verbose=False  # Keep logs clean
    )

    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f}s\n")

    # 3. Prepare Dataset
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print(f"🚀 Starting Inference Stream on {len(files)} files...\n")

    for i, path in enumerate(files):
        fname = os.path.basename(path)

        try:
            # Read PDF
            # Note: Ensure 'pypdf' is installed in your setup script!
            reader = PdfReader(path)
            if len(reader.pages) > 0:
                text = reader.pages[0].extract_text()[:3500]
            else:
                text = ""

            # Inference (Loop)
            start_time = time.time()

            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},  # Hardware-constrained JSON
                temperature=0.1
            )

            # Metrics
            inference_time = time.time() - start_time
            content = resp['choices'][0]['message']['content']
            tokens = resp['usage']['completion_tokens']
            tps = tokens / inference_time if inference_time > 0 else 0

            # Parse Data
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {}

            # Save Row
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
                "tokens": tokens,
                "mode": "warm",
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

            # Progress Bar (Every 10 files)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f} | Time: {inference_time:.2f}s")

            # CRITICAL: Soft Reset KV Cache
            # LFM is state-sensitive. This helps prevent "State Drift".
            llm.reset()

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

    # 4. Cleanup & Save
    del llm

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    output_csv = os.path.join(LOG_DIR, "h100_lfm_baseline_warm.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # 5. Summary
    successful = df[df['tps'] > 0]
    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPLETE")
    print(f"Saved to: {output_csv}")
    if len(successful) > 0:
        print(f"Mean TPS: {successful['tps'].mean():.2f}")
        print(f"Max TPS:  {successful['tps'].max():.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_lfm_warm()