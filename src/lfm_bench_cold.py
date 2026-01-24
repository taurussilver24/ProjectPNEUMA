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
MODEL_FILE = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"  # Corrected filename
FILE_LIMIT = 5  # Set to 1000 for your marathon

# --- 2. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join("/run/media/taurus/Games/models/", MODEL_FILE)
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# SYSTEM_PROMPT = """You are a metadata extraction system. Extract fields into JSON:
# { "title": "", "authors": [], "doi": "", "arxiv_id": "", "keywords": [], "summary": "" }
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

def run_lfm_marathon():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: LFM model not found at {MODEL_PATH}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print(f"\n--- 🚀 STARTING LFM-2.5 COLD-START MARATHON ({len(files)} files) ---")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        llm = None

        try:
            # 1. READ PDF
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # 2. COLD START: Re-initialize for every file to fix -1 crash
            start_time = time.time()
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False
            )

            # 3. INFERENCE
            resp = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            data = json.loads(resp['choices'][0]['message']['content'])
            tokens = resp['usage']['completion_tokens']
            tps = tokens / (time.time() - start_time)

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
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

            print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f} (Cold Start)")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

        finally:
            # 4. MEMORY PURGE: Explicitly kill the model instance and flush VRAM
            if llm:
                del llm
            gc.collect()
            # Note: The RX 6800 Vulkan driver will now fully release the VRAM buffer

    output_csv = os.path.join(LOG_DIR, f"clash_LFM-2.5_{PLATFORM_NAME.replace(' ', '_')}.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n✅ LFM Marathon Complete. Results: {output_csv}")


if __name__ == "__main__":
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    run_lfm_marathon()