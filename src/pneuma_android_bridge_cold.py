import os
import time
import json
import re
import glob
import subprocess
import pandas as pd
from pypdf import PdfReader

# --- CONFIGURATION ---Qwen2.5-3B  LFM2.5-1.2B
PLATFORM_NAME = "Dimensity_9300+(Immortalis Vulkan)"
MODEL_NAME = "Phi-4"
FILE_LIMIT = 1000
LOG_FILE_NAME = f"clash_{MODEL_NAME}_final.csv"

# Paths
ADB_PATH = "/home/taurus/Android/Sdk/platform-tools/adb"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CSV_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

DEVICE_DIR = "/data/local/tmp"
DEVICE_MODEL_PATH = "/data/local/tmp/phi-4-q4_k_m.gguf"
DEVICE_PROMPT_FILE = "current_prompt.txt"

SYSTEM_PROMPT = """You are a metadata extraction system. Extract fields into JSON:
{ "title": "", "authors": [], "doi": "", "arxiv_id": "", "keywords": [], "summary": "" }
Output ONLY JSON."""

def parse_metrics_from_log(logs, total_time):
    gen_match = re.search(r'Generation:\s+(\d+\.\d+)\s+t/s', logs)
    tps = float(gen_match.group(1)) if gen_match else 0.0
    return tps, int(tps * total_time)

def clean_text_for_json(raw_output):
    if not raw_output: return None
    text = raw_output.replace(" || ", "\n")
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    candidates = []
    for m in matches:
        try:
            blob = m.group(0)
            data = json.loads(blob)
            if data.get("title", "").strip() or data.get("summary", "").strip():
                candidates.append(data)
        except:
            continue
    return candidates[-1] if candidates else None

def run_android_benchmark():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    # --- MINIMAL RESUME LOGIC ---
    processed_files = set()
    results = []
    if os.path.exists(CSV_PATH):
        try:
            existing_df = pd.read_csv(CSV_PATH)
            results = existing_df.to_dict('records')
            processed_files = set(existing_df['filename'].tolist())
            print(f"🔄 RESUME MODE: {len(processed_files)} files already processed.")
        except:
            pass

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]
    print("\n" + "=" * 70)
    print(f"🚀 PNEUMA RUN: {MODEL_NAME} on {PLATFORM_NAME}")
    print("=" * 70 + "\n")

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        if fname in processed_files: continue

        print(f"--- [{i + 1}/{len(files)}] Processing: {fname} ---")

        try:
            # 1. READ PDF (Edit: Added encoding cleanup to strip math symbols/surrogates)
            print("   📄 Reading PDF...", end="", flush=True)
            try:
                reader = PdfReader(path)
                raw_text = reader.pages[0].extract_text() or ""
                text = raw_text.encode("utf-8", "ignore").decode("utf-8").replace('\x00', '')[:3000]
                print(f" Done ({len(text)} chars)")
            except Exception as e:
                print(f" ❌ ERROR: {e}")
                continue

            # 2. PUSH PROMPT
            print("   📲 Pushing...", end="", flush=True)
            full_prompt = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

            with open("temp_prompt.txt", "w", encoding="utf-8") as f:
                f.write(full_prompt)
            subprocess.run([ADB_PATH, "push", "temp_prompt.txt", f"{DEVICE_DIR}/{DEVICE_PROMPT_FILE}"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(" Done")

            # 3. EXECUTE (Edit: Added errors='replace' to prevent UTF-8 crashes on live logs)
            print("   🔥 Immortalis Live Feed:")
            cmd = (
                f'{ADB_PATH} shell "cd {DEVICE_DIR} && '
                f'LD_LIBRARY_PATH={DEVICE_DIR}:/vendor/lib64:/system/lib64 '
                f'./pneuma_engine -m {DEVICE_MODEL_PATH} '
                f'-f {DEVICE_PROMPT_FILE} '
                f'-n 512 -c 2048 -ngl 99 -t 8 --temp 0 --simple-io < /dev/null"'
            )

            start_time = time.time()
            full_log_buffer = []

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace' # <--- MINIMAL FIX
            )

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    clean_line = line.strip()
                    full_log_buffer.append(clean_line)

                    if "t/s" in clean_line:
                        print(f"      [STATS] {clean_line}")
                        process.kill()
                        break
                if time.time() - start_time > 300:
                    process.kill()
                    print("      ❌ TIMEOUT KILL")
                    break

            # 4. PARSE & SAVE
            total_time = time.time() - start_time
            full_output_str = "\n".join(full_log_buffer)
            tps, _ = parse_metrics_from_log(full_output_str, total_time)
            data = clean_text_for_json(full_output_str)

            def safe_join(val):
                if isinstance(val, list): return ", ".join(str(x) for x in val)
                return str(val) if val else "N/A"

            results.append({
                "filename": fname,
                "title": data.get("title", "N/A") if data else "N/A",
                "authors": safe_join(data.get("authors", [])) if data else "N/A",
                "doi": data.get("doi", "N/A") if data else "N/A",
                "arxiv_id": data.get("arxiv_id", "N/A") if data else "N/A",
                "keywords": safe_join(data.get("keywords", [])) if data else "N/A",
                "summary": data.get("summary", "N/A") if data else "N/A",
                "tps": tps,
                "model": MODEL_NAME,
                "platform": PLATFORM_NAME,
                "raw_output": full_output_str.replace('\n', ' || ')
            })

            pd.DataFrame(results).to_csv(CSV_PATH, index=False)
            print(f"   💾 SAVED. (Rows: {len(results)})")
            time.sleep(2) # Thermal Interleaving

        except Exception as e:
            print(f"\n   ❌ CRASH: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_android_benchmark()