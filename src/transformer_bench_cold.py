import time
import json
import os
import glob
import gc
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- 1. RESEARCH CONFIGURATION ---
PLATFORM_NAME = "AMD RX 6800 (Vulkan)"
FILE_LIMIT = 5

# --- 2. MODEL QUEUE ---
MODEL_QUEUE = [
    {"name": "Qwen2.5-3B", "file": "qwen2.5-3b-instruct-q4_k_m.gguf"},
    {"name": "Phi-4", "file": "phi-4-q4_k_m.gguf"}
]

# --- 3. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = "/run/media/taurus/Games/models/"
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# --- 4. ENHANCED SYSTEM PROMPT (Restored Fields) ---
# SYSTEM_PROMPT = """You are a metadata extraction system. Extract fields into JSON:
# {
#     "title": "Exact paper title",
#     "authors": ["List of names"],
#     "doi": "DOI string if found",
#     "arxiv_id": "ID if found",
#     "keywords": ["Technical tags"],
#     "summary": "1-sentence abstract"
# }
# Output ONLY JSON."""

# SYSTEM_PROMPT = """You are a high-precision academic metadata extractor.
# Your task is to parse the provided text and extract specific metadata fields into a strict JSON format.
#
# RULES:
# 1. Output ONLY valid JSON. Do not include markdown formatting (like ```json).
# 2. If a field is not found, use null (do not make up data).
# 3. "doi" must start with '10.'.
# 4. "arxiv_id" must match the pattern 'YYMM.NNNNN'.
#
# REQUIRED JSON STRUCTURE:
# {
#     "title": "Exact title of the paper",
#     "authors": ["Author 1", "Author 2", ...],
#     "doi": "10.xxxx/xxxxx or null",
#     "arxiv_id": "2401.12345 or null",
#     "keywords": ["Key technical term 1", "Key technical term 2", ...],
#     "summary": "A concise, single-sentence summary of the main contribution."
# }"""



SYSTEM_PROMPT = """You are an expert research librarian. Analyze the document to extract metadata.

STEP 1: REASONING
First, think silently about the document structure:
- Identify the main title (usually largest font/first page).
- Distinguish the actual authors from affiliations.
- Locate the specific DOI of *this* paper, ignoring DOIs in the references section.
- Synthesize the core contribution for the summary.

STEP 2: EXTRACTION
After your reasoning, output the final metadata in this exact JSON format inside a code block:

```json
{
    "title": "Exact string",
    "authors": ["List of strings"],
    "doi": "String or null",
    "arxiv_id": "String or null",
    "keywords": ["List of strings"],
    "summary": "String"
}"""


def run_cold_benchmark(model_config):
    """Run cold-start benchmark for a single model"""
    model_path = os.path.join(MODEL_DIR, model_config["file"])

    if not os.path.exists(model_path):
        print(f"❌ Error: {model_config['name']} not found at {model_path}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print("\n" + "=" * 70)
    print(f"🚀 STARTING {model_config['name']} COLD-START BENCHMARK")
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Files: {len(files)}")
    print("=" * 70 + "\n")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        llm = None

        try:
            # 1. READ PDF
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # 2. COLD START: Re-initialize for every file
            start_time = time.time()
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
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

            data = json.loads(resp['choices'][0]['message']['content'])
            tokens = resp['usage']['completion_tokens']
            total_time = time.time() - start_time
            tps = tokens / total_time

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
                "total_time": round(total_time, 3),
                "tokens": tokens,
                "mode": "cold_start",
                "model": model_config["name"],
                "platform": PLATFORM_NAME
            })

            # Progress output (every 10th file)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f} | Time: {total_time:.2f}s")

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
                "total_time": 0,
                "tokens": 0,
                "mode": "cold_start",
                "model": model_config["name"],
                "platform": PLATFORM_NAME
            })

        finally:
            # 4. MEMORY PURGE
            if llm:
                del llm
            gc.collect()

    # Save results
    output_csv = os.path.join(LOG_DIR, f"clash_{model_config['name']}_cold_start_{PLATFORM_NAME.replace(' ', '_')}.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # Summary statistics
    successful = df[df['tps'] > 0]

    print("\n" + "=" * 70)
    print(f"📊 {model_config['name']} COLD-START BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\n✅ Results saved to: {output_csv}\n")

    if len(successful) > 0:
        print(f"Successful extractions: {len(successful)}/{len(files)}")
        print(f"\nPerformance Metrics:")
        print(f"  Mean TPS:             {successful['tps'].mean():.2f}")
        print(f"  Median TPS:           {successful['tps'].median():.2f}")
        print(f"  Min TPS:              {successful['tps'].min():.2f}")
        print(f"  Max TPS:              {successful['tps'].max():.2f}")
        print(f"  Std Dev:              {successful['tps'].std():.2f}")
        print(f"\n  Mean total time:      {successful['total_time'].mean():.2f}s")
        print(f"  Total runtime:        {successful['total_time'].sum():.2f}s")
    else:
        print("⚠️ No successful extractions")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # Run cold-start benchmark for each Transformer model
    for model in MODEL_QUEUE:
        run_cold_benchmark(model)

    print("\n" + "=" * 70)
    print("🎯 ALL TRANSFORMER COLD-START BENCHMARKS COMPLETE")
    print("=" * 70)
    print("\nCSV files generated:")
    for model in MODEL_QUEUE:
        csv_name = f"clash_{model['name']}_cold_start_{PLATFORM_NAME.replace(' ', '_')}.csv"
        print(f"  - logs/{csv_name}")
    print("\n" + "=" * 70 + "\n")