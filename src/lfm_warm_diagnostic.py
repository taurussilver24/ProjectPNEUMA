import time
import json
import os
import glob
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- CONFIGURATION ---
PLATFORM_NAME = "AMD RX 6800 (Vulkan)"
MODEL_FILE = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
FILE_LIMIT = 1000

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "/run/media/taurus/Games/models/" + MODEL_FILE
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# --- SYSTEM PROMPT ---
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
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    print("\n" + "=" * 70)
    print(f"🔥 LFM-2.5 WARM MODE BENCHMARK")
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Files: {FILE_LIMIT}")
    print("=" * 70 + "\n")

    # LOAD MODEL ONCE (WARM MODE)
    print("Loading LFM-2.5 model (this happens ONCE)...")
    start_load = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=512,
        verbose=False
    )
    load_time = time.time() - start_load
    print(f"✅ Model loaded in {load_time:.2f}s\n")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print(f"Processing {len(files)} files in WARM mode...\n")

    for i, path in enumerate(files):
        fname = os.path.basename(path)

        try:
            # Read PDF
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # Inference (model stays loaded)
            start_time = time.time()
            resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            data = json.loads(resp['choices'][0]['message']['content'])
            tokens = resp['usage']['completion_tokens']
            inference_time = time.time() - start_time
            tps = tokens / inference_time

            # CRITICAL: Clear KV cache after each file to prevent corruption
            try:
                llm.reset()  # Flush KV cache but keep model loaded
            except AttributeError:
                pass  # Method not available in this llama-cpp version

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A"),
                "authors": ", ".join(data.get("authors", [])) if isinstance(data.get("authors"), list) else data.get(
                    "authors", "N/A"),
                "doi": data.get("doi", "None"),
                "arxiv_id": data.get("arxiv_id", "None"),
                "keywords": ", ".join(data.get("keywords", [])) if isinstance(data.get("keywords"), list) else data.get(
                    "keywords", "None"),
                "summary": data.get("summary", "N/A"),
                "tps": round(tps, 2),
                "inference_time": round(inference_time, 3),
                "tokens": tokens,
                "mode": "warm",
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

            # Progress output
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f} | Time: {inference_time:.2f}s")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")
            results.append({
                "filename": fname,
                "title": "ERROR",
                "authors": "ERROR",
                "doi": "ERROR",
                "arxiv_id": "ERROR",
                "keywords": "ERROR",
                "summary": str(e),
                "tps": 0,
                "inference_time": 0,
                "tokens": 0,
                "mode": "warm",
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

    # Cleanup
    del llm

    # Save results
    output_csv = os.path.join(LOG_DIR, f"clash_LFM-2.5_warm_{PLATFORM_NAME.replace(' ', '_')}.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Summary Statistics
    successful = df[df['tps'] > 0]

    print("\n" + "=" * 70)
    print("📊 WARM MODE BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\n✅ Results saved to: {output_csv}\n")

    if len(successful) > 0:
        print(f"Successful extractions: {len(successful)}/{len(files)}")
        print(f"Model load time:        {load_time:.2f}s")
        print(f"\nPerformance Metrics:")
        print(f"  Mean TPS:             {successful['tps'].mean():.2f}")
        print(f"  Median TPS:           {successful['tps'].median():.2f}")
        print(f"  Min TPS:              {successful['tps'].min():.2f}")
        print(f"  Max TPS:              {successful['tps'].max():.2f}")
        print(f"  Std Dev:              {successful['tps'].std():.2f}")
        print(f"\n  Mean inference time:  {successful['inference_time'].mean():.2f}s")
        print(f"  Total runtime:        {successful['inference_time'].sum():.2f}s")

        # Compare to cold start (if exists)
        cold_csv = os.path.join(LOG_DIR, f"clash_LFM-2.5_{PLATFORM_NAME.replace(' ', '_')}.csv")
        if os.path.exists(cold_csv):
            cold_df = pd.read_csv(cold_csv)
            cold_successful = cold_df[cold_df['tps'] > 0]
            if len(cold_successful) > 0:
                cold_tps = cold_successful['tps'].mean()
                warm_tps = successful['tps'].mean()
                speedup = ((warm_tps - cold_tps) / cold_tps) * 100
                print(f"\n🔥 WARM vs COLD START COMPARISON:")
                print(f"  Cold start TPS:       {cold_tps:.2f}")
                print(f"  Warm mode TPS:        {warm_tps:.2f}")
                print(f"  Speedup:              {speedup:+.1f}%")
    else:
        print("⚠️ No successful extractions")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    run_lfm_warm()