import time
import json
import os
import glob
import pandas as pd
import requests
from pypdf import PdfReader

# --- CONFIG ---
PLATFORM_NAME = "Windows RTX 4080 (Native CUDA)"
API_URL = "http://localhost:8080/v1/chat/completions"
# Set this to 1000 for the full run
FILE_LIMIT = 1000

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset",)
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")

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


def run_standard_benchmark():
    # 1. Setup
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    print(f"\n" + "=" * 60)
    print(f"🔥 LFM-2.5 WARM MODE BENCHMARK")
    print(f"Platform: {PLATFORM_NAME}")
    print(f"Files: {len(files)}")
    print("=" * 60 + "\n")

    # 2. Check Server
    try:
        requests.get("http://localhost:8080/health")
    except:
        print("❌ CRITICAL: llama-server.exe is not running!")
        return

    results = []

    # 3. Processing Loop
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            # READ PDF
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # PAYLOAD
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }

            # INFERENCE
            resp = requests.post(API_URL, json=payload).json()

            # --- METRIC CALCULATION (The Fair Comparison) ---
            timings = resp.get('timings', {})

            # Write Speed (Generation only)
            g_n = timings.get('predicted_n', 0)
            g_ms = timings.get('predicted_ms', 0)

            if g_ms > 0:
                write_tps = (g_n / g_ms) * 1000
            else:
                write_tps = 0

            # --- PARSING ---
            content = resp['choices'][0]['message']['content']

            # Clean Markdown wrappers
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]

            try:
                data = json.loads(content)
            except:
                data = {}

            # Formatting
            title = data.get("title", "N/A")
            authors = data.get("authors", [])
            keywords = data.get("keywords", [])
            authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
            keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)

            print(f"[{i + 1}/{len(files)}] {fname[:15]}... | TPS: {write_tps:.2f} | Title: {title[:30]}...")

            results.append({
                "filename": fname,
                "title": title,
                "authors": authors_str,
                "doi": data.get("doi", "None"),
                "arxiv_id": data.get("arxiv_id", "None"),
                "keywords": keywords_str,
                "summary": data.get("summary", "N/A"),
                "tps": round(write_tps, 2),  # <--- THIS IS NOW PURE WRITE SPEED
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

    # 4. SAVE (Standard Filename)
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    # Matches AMD naming convention
    output_csv = os.path.join(LOG_DIR, "clash_LFM-2.5_warm_Nvidia_RTX_4080(CUDA).csv")

    cols = ["filename", "title", "authors", "doi", "arxiv_id", "keywords", "summary", "tps", "model", "platform"]
    pd.DataFrame(results)[cols].to_csv(output_csv, index=False)

    print(f"\n✅ Results saved: {output_csv}")


if __name__ == "__main__":
    run_standard_benchmark()