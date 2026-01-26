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

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: No '.exe' for Linux
SERVER_EXE = os.path.join(BASE_DIR, "..", "llama_dist", "llama-server")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")
API_URL = "http://localhost:8080/v1/chat/completions"

# --- PROMPT (Schema Mode - No Examples) ---
# We use this prompt to force the model to generate tokens,
# proving the B200's bandwidth speed.
SYSTEM_PROMPT = """You are a research librarian. Extract metadata from the provided academic paper.

INSTRUCTIONS:
1. Read the user text carefully.
2. Generate a VALID JSON object based on the schema below.
3. Do NOT output markdown or explanations. Just the JSON.

REQUIRED JSON FIELDS:
- "title": (String) The actual title of the paper.
- "authors": (List of Strings) The names of the authors.
- "doi": (String) The DOI if found, else null.
- "arxiv_id": (String) The ArXiv ID if found, else null.
- "keywords": (List of Strings) Key terms.
- "summary": (String) A one-sentence summary.
"""


def wait_for_server():
    """Polls server tight."""
    for _ in range(200):  # 10 seconds max
        try:
            if requests.get("http://localhost:8080/health", timeout=0.5).status_code == 200:
                return True
        except:
            pass
        time.sleep(0.05)
    return False


def run_b200_cold_benchmark():
    print(f"\n" + "=" * 60)
    print(f"🧊 LFM-2.5 COLD MODE (B200 HARD RESET)")
    print(f"Platform: {PLATFORM_NAME}")
    print(f"=" * 60)

    if not os.path.exists(SERVER_EXE):
        print(f"❌ CRITICAL: Server not found at {SERVER_EXE}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        server_process = None

        try:
            # 1. START TIMER (Real Wall Clock)
            # On B200 Linux, "Boot" is part of the performance metric.
            start_time = time.time()

            # 2. BOOT SERVER
            # -ngl 99: Force all layers to GPU
            server_process = subprocess.Popen(
                [SERVER_EXE, "-m", MODEL_PATH, "-ngl", "99", "-c", "4096", "--port", "8080", "--host", "0.0.0.0"],
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
            # We want to know: "How long from 0 to Answer?"
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
                "summary": data.get("summary", "N/A"),
                "tps": round(tps, 2),
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

        finally:
            # 7. HARD KILL (Linux)
            # Ensure the process is dead before next loop
            if server_process:
                server_process.terminate()
            subprocess.run(["pkill", "-9", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 8. SAVE
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    csv_name = "clash_LFM-2.5_Cold_Nvidia_B200.csv"
    pd.DataFrame(results).to_csv(os.path.join(LOG_DIR, csv_name), index=False)
    print(f"\n✅ Saved: {csv_name}")


if __name__ == "__main__":
    run_b200_cold_benchmark()