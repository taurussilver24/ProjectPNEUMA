import time
import json
import os
import glob
import pandas as pd
import requests
import subprocess
import signal
from pypdf import PdfReader

# --- CONFIG ---
PLATFORM_NAME = "Windows RTX 4080 (Native CUDA)"
# Correct path to your llama_dist folder
SERVER_EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama_dist", "llama-server.exe")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models",
                          "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")
API_URL = "http://localhost:8080/v1/chat/completions"
FILE_LIMIT = 5  # Set to 1000 for marathon

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dataset")
LOG_DIR = os.path.join(BASE_DIR, "..", "logs")

# --- ANTI-PARROT PROMPT ---
SYSTEM_PROMPT = """You are an expert research librarian. 
Your task is to extract metadata from the academic paper provided by the user.

OUTPUT FORMAT:
Return ONLY a valid JSON object. 
Do NOT copy any example text. Extract real data from the document.

{
    "title": "Extract the full title here",
    "authors": ["List of authors here"],
    "doi": "Extract DOI or null",
    "arxiv_id": "Extract ArXiv ID or null",
    "keywords": ["Extract keywords here"],
    "summary": "Write a 1-sentence summary here"
}"""


def wait_for_server():
    """Polls tight to minimize latency penalty."""
    retries = 0
    while retries < 200:  # Wait up to 10 seconds
        try:
            resp = requests.get("http://localhost:8080/health", timeout=0.5)
            if resp.status_code == 200:
                return True
        except:
            pass
        # High precision sleep (50ms) so we don't punish the benchmark
        time.sleep(0.05)
        retries += 1
    return False


def run_true_cold_benchmark():
    # 1. Validation
    if not os.path.exists(SERVER_EXE):
        print(f"❌ Error: Could not find server at {SERVER_EXE}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    results = []

    print(f"\n--- 🧊 STARTING TRUE COLD START BENCHMARK (Server Restart) ---")
    print(f"    Platform: {PLATFORM_NAME}")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        server_process = None

        try:
            # 2. START TIMER (Matches AMD Logic: Init + Inference)
            start_time = time.time()

            # 3. LAUNCH SERVER (Model Load Penalty)
            # -ngl 99 forces ALL layers to GPU immediately
            server_process = subprocess.Popen(
                [SERVER_EXE, "-m", MODEL_PATH, "-ngl", "99", "-c", "4096", "--port", "8080"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # 4. WAIT FOR IGNITION
            if not wait_for_server():
                print(f"❌ Server failed to start for {fname}")
                continue

            # 5. READ PDF
            text = PdfReader(path).pages[0].extract_text()[:3500]

            # 6. INFERENCE
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }

            resp = requests.post(API_URL, json=payload).json()

            # 7. METRICS (Strict Wall Clock)
            end_time = time.time()
            total_duration = end_time - start_time

            tokens = resp['usage']['completion_tokens']
            # This TPS includes the time it took to boot the server
            tps = tokens / total_duration

            # 8. PARSING
            content = resp['choices'][0]['message']['content']
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]

            try:
                data = json.loads(content)
                title = data.get("title", "N/A")
                if "Extract" in title: title = "CHEAT_DETECTED"
            except:
                data = {}
                title = "JSON_FAIL"

            print(f"[{i + 1}/{len(files)}] {fname[:15]}... | Cold TPS: {tps:.2f} | Title: {title[:30]}...")

            results.append({
                "filename": fname,
                "title": title,
                "authors": str(data.get("authors", [])),
                "doi": data.get("doi", "None"),
                "arxiv_id": data.get("arxiv_id", "None"),
                "keywords": str(data.get("keywords", [])),
                "summary": data.get("summary", "N/A"),
                "tps": round(tps, 2),
                "model": "LFM-2.5",
                "platform": PLATFORM_NAME
            })

        except Exception as e:
            print(f"❌ Error on {fname}: {e}")

        finally:
            # 9. KILL SERVER (Reset for next file)
            if server_process:
                # Force kill to ensure port 8080 is freed instantly
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(server_process.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 10. SAVE
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    output_csv = os.path.join(LOG_DIR, f"clash_LFM-2.5_Cold_Nvidia_RTX_4080(CUDA).csv")
    cols = ["filename", "title", "authors", "doi", "arxiv_id", "keywords", "summary", "tps", "model", "platform"]
    pd.DataFrame(results)[cols].to_csv(output_csv, index=False)
    print(f"\n✅ Cold Results: {output_csv}")


if __name__ == "__main__":
    run_true_cold_benchmark()