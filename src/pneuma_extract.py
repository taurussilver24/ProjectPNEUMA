import time
import json
import os
import glob
import re
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- 1. RESEARCH CONFIGURATION ---
PLATFORM_NAME = "Nvidia RTX 4080M(CUDA)"  # Switch to "Apple M1 (Metal)" on Mac
FILE_LIMIT = 5  # Set to 15 for your pilot test
MODE = "SLM"  # "SLM" or "REGEX"

# --- 2. MULTI-MODEL QUEUE ---
MODEL_QUEUE = [
    {"name": "Qwen2.5-3B", "file": "qwen2.5-3b-instruct-q4_k_m.gguf"},
    {"name": "Phi-4", "file": "phi-4-q4_k_m.gguf"}
]

# --- 3. DYNAMIC PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_PATHS = [
    "../models/",  # Linux/W11 Desktop Path
    "/Volumes/Games/models/"  # M1 Mac Path
]


def resolve_model_dir():
    for path in EXTERNAL_PATHS:
        if os.path.exists(path): return path
    return os.path.join(BASE_DIR, "models")


MODEL_DIR = resolve_model_dir()
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


# --- 5. LOGIC ENGINE ---
def regex_extract(text):
    doi = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.I)
    arxiv = re.search(r'arXiv:\d{4}\.\d{4,5}', text)
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 15]
    title = next((l for l in lines[:5] if not re.search(r'(arXiv|downloaded|journal|http)', l, re.I)), "Not Found")
    return {
        "title": title,
        "authors": "REGEX_NER_PENDING",
        "doi": doi.group(0) if doi else "None",
        "arxiv_id": arxiv.group(0) if arxiv else "None",
        "keywords": "None",
        "summary": "None"
    }


def run_tier(model_config):
    model_path = os.path.join(MODEL_DIR, model_config["file"])
    if not os.path.exists(model_path) and MODE == "SLM":
        print(f"⚠️ Skipping {model_config['name']}: Not found at {model_path}")
        return

    print(f"\n--- 🏁 STARTING TIER: {model_config['name']} ---")

    # Initialize LLM (Auto-detects backend based on llama-cpp-python installation)
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False) if MODE == "SLM" else None

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            reader = PdfReader(path)
            text = reader.pages[0].extract_text()[:3500]
            start_time = time.time()

            if MODE == "SLM":
                resp = llm.create_chat_completion(
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                data = json.loads(resp['choices'][0]['message']['content'])
                completion_tokens = resp['usage']['completion_tokens']
                tps = completion_tokens / (time.time() - start_time)
            else:
                data = regex_extract(text)
                tps = 0

            # --- CAPTURING ALL FIELDS FOR THE CLASH ---
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
                "model": model_config["name"],
                "platform": PLATFORM_NAME
            })
            print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f}")

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

    output_csv = os.path.join(LOG_DIR, f"clash_{model_config['name']}_{PLATFORM_NAME.replace(' ', '_')}.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"✅ Saved results to: {output_csv}")


if __name__ == "__main__":
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    if MODE == "REGEX":
        run_tier({"name": "REGEX_BASELINE", "file": ""})
    else:
        for m in MODEL_QUEUE:
            run_tier(m)


