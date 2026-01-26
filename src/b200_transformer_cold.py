import time
import json
import os
import glob
import pandas as pd
import requests
import subprocess
from pypdf import PdfReader

# --- 1. CONFIGURATION ---
PLATFORM_NAME = "Nvidia B200 (Cloud Native)"
FILE_LIMIT = 1000  # Go all in

# --- 2. MODEL QUEUE ---
MODEL_QUEUE = [
    {"name": "Qwen2.5-3B", "file": "qwen2.5-3b-instruct-q4_k_m.gguf"},
    {"name": "Phi-4", "file": "phi-4-q4_k_m.gguf"}
]

# --- 3. PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: Linux binary (no .exe)
SERVER_EXE = os.path.join(BASE_DIR, "..", "llama_dist", "llama-server")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
API_URL = "http://localhost:8080/v1/chat/completions"

# --- 4. EXACT PROMPT FROM AMD RUN ---
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
    for _ in range(200):  # 10 seconds max
        try:
            if requests.get("http://localhost:8080/health", timeout=0.5).status_code == 200:
                return True
        except:
            pass
        time.sleep(0.05)
    return False


def run_cold_tier(model_config):
    model_path = os.path.join(MODELS_DIR, model_config["file"])
    if not os.path.exists(model_path):
        print(f"⚠️ Skipping {model_config['name']}: Not found at {model_path}")
        return

    print(f"\n" + "=" * 70)
    print(f"🧊 STARTING COLD TIER: {model_config['name']}")
    print(f"   Platform: {PLATFORM_NAME}")
    print(f"=" * 70)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        server_process = None

        try:
            # 1. START TIMER (Real Wall Clock: Boot + Inference)
            # This matches your AMD logic: Time starts when we ask to load the model.
            start_time = time.time()

            # 2. BOOT SERVER (Simulates loading library)
            # -ngl 99: Force B200
            server_process = subprocess.Popen(
                [SERVER_EXE, "-m", model_path, "-ngl", "99", "-c", "4096", "--port", "8080", "--host", "0.0.0.0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # 3. WAIT FOR IGNITION
            if not wait_for_server():
                print(f"❌ Server failed to start for {fname}")
                continue

            # 4. READ & INFER
            text = PdfReader(path).pages[0].extract_text()[:3500]

            payload = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
                "temperature": 0.1, "response_format": {"type": "json_object"}
            }

            resp = requests.post(API_URL, json=payload).json()

            # 5. METRICS (Strict Total Time)
            end_time = time.time()
            total_duration = end_time - start_time

            tokens = resp['usage']['completion_tokens']
            tps = tokens / total_duration

            # 6. PARSING
            content = resp['choices'][0]['message']['content']
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]

            try:
                data = json.loads(content)
                title = data.get("title", "N/A")
            except:
                data = {}
                title = "JSON_FAIL"

            print(
                f"[{i + 1}/{len(files)}] {fname[:10]}... | Time: {total_duration:.2f}s | Cold TPS: {tps:.2f} | Title: {title[:20]}...")

            results.append({
                "filename": fname,
                "title": title,
                "authors": str(data.get("authors", [])),
                "doi": data.get("doi", "None"),
                "arxiv_id": data.get("arxiv_id", "None"),
                "keywords": str(data.get("keywords", [])),
                "summary": data.get("summary", "N/A"),
                "tps": round(tps, 2),
                "total_time": round(total_duration, 3),
                "mode": "cold_start",
                "model": model_config["name"],
                "platform": PLATFORM_NAME
            })

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

        finally:
            # 7. HARD KILL (Linux)
            if server_process:
                server_process.terminate()
            subprocess.run(["pkill", "-9", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Short cooldown to ensure port release
            time.sleep(0.1)

    # 8. SAVE
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    csv_name = f"clash_{model_config['name']}_Cold_Nvidia_B200.csv"
    pd.DataFrame(results).to_csv(os.path.join(LOG_DIR, csv_name), index=False)
    print(f"\n✅ Saved: {csv_name}")


if __name__ == "__main__":
    for m in MODEL_QUEUE:
        run_cold_tier(m)