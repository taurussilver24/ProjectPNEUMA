import time
import json
import os
import glob
import pandas as pd
import requests
import subprocess
from pypdf import PdfReader

# --- CONFIG ---
PLATFORM_NAME = "Nvidia B200 (Cloud Native)"
FILE_LIMIT = 1000  # Full run

# --- PATHS (Linux/Cloud Structure) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: No '.exe' for Linux binary
SERVER_EXE = os.path.join(BASE_DIR, "..", "llama_dist", "llama-server")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
API_URL = "http://localhost:8080/v1/chat/completions"

# --- PROMPT ---
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


def wait_for_server():
    """Polls server until ready."""
    print("⏳ Waiting for B200 Ignition...")
    for _ in range(200):  # 10 seconds max
        try:
            if requests.get("http://localhost:8080/health", timeout=0.5).status_code == 200:
                return True
        except:
            pass
        time.sleep(0.05)
    return False


def run_b200_warm_benchmark():
    print(f"\n" + "=" * 60)
    print(f"🔥 LFM-2.5 WARM MODE (B200 EDITION)")
    print(f"Platform: {PLATFORM_NAME}")
    print(f"=" * 60 + "\n")

    if not os.path.exists(SERVER_EXE):
        print(f"❌ CRITICAL: Could not find server binary at {SERVER_EXE}")
        print("   Did you run ./setup_b200.sh?")
        return

    # 1. KILL OLD SERVERS (Linux Command)
    subprocess.run(["pkill", "-9", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)

    # 2. START SERVER
    # -ngl 99: Force all layers to GPU
    # -c 32768: Large context window
    print(f"🚀 Launching Engine...")
    server_process = subprocess.Popen(
        [SERVER_EXE, "-m", MODEL_PATH, "-ngl", "99", "-c", "32768", "--port", "8080", "--host", "0.0.0.0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    if not wait_for_server():
        print("❌ Server failed to start! Check logs.")
        server_process.terminate()
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    # 3. PROCESSING LOOP
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        try:
            text = PdfReader(path).pages[0].extract_text()[:3500]

            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }

            resp = requests.post(API_URL, json=payload).json()

            # METRICS (Write Speed)
            timings = resp.get('timings', {})
            g_n = timings.get('predicted_n', 0)
            g_ms = timings.get('predicted_ms', 0)
            write_tps = (g_n / g_ms * 1000) if g_ms > 0 else 0

            # PARSING
            content = resp['choices'][0]['message']['content']
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]

            try:
                data = json.loads(content)
            except:
                data = {}

            title = data.get("title", "N/A")
            print(f"[{i + 1}/{len(files)}] {fname[:10]}... | TPS: {write_tps:.2f} | Title: {title[:20]}...")

            results.append({
                "filename": fname,
                "title": title,
                "authors": str(data.get("authors", [])),
                "summary": data.get("summary", "N/A"),
                "tps": round(write_tps, 2),
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

    # 4. CLEANUP & SAVE
    server_process.terminate()
    # Force kill just in case
    subprocess.run(["pkill", "-9", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    # Filename matches AMD/Nvidia convention
    output_csv = os.path.join(LOG_DIR, "clash_LFM-2.5_Warm_Nvidia_B200.csv")

    cols = ["filename", "title", "authors", "summary", "tps", "model", "platform"]
    pd.DataFrame(results)[cols].to_csv(output_csv, index=False)
    print(f"\n✅ Results saved: {output_csv}")


if __name__ == "__main__":
    run_b200_warm_benchmark()