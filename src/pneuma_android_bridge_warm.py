# MODEL_NAME = "LFM-2.5"  # CHANGE THIS toQwen2.5-3B "LFM-2.5" or "Phi-4"

import os
import time
import json
import re
import glob
import subprocess
import pandas as pd
from pypdf import PdfReader
import logging

# --- CONFIGURATION ---
PLATFORM_NAME = "Dimensity_9300+(Immortalis Vulkan)"
MODEL_NAME = "Qwen"
FILE_LIMIT = 1000

LOG_FILE_NAME = f"clash_{MODEL_NAME}_warm_Dimensity.csv"

# Paths
ADB_PATH = "/home/taurus/Android/Sdk/platform-tools/adb"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")
CSV_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

DEVICE_DIR = "/data/local/tmp"
DEVICE_DATASET_DIR = f"{DEVICE_DIR}/pneuma_dataset"

if "Qwen" in MODEL_NAME:
    DEVICE_MODEL_PATH = "/data/local/tmp/qwen2.5-3b-instruct-q4_k_m.gguf"
elif "Phi" in MODEL_NAME:
    DEVICE_MODEL_PATH = "/data/local/tmp/phi-4-q4_k_m.gguf"
elif "LFM" in MODEL_NAME:
    DEVICE_MODEL_PATH = "/data/local/tmp/lfm.gguf"

SYSTEM_PROMPT = """You are a metadata extraction system. Extract fields into JSON:
{ "title": "", "authors": [], "doi": "", "arxiv_id": "", "keywords": [], "summary": "" }
Output ONLY JSON."""


def setup_device_dataset():
    print("📦 BULK UPLOAD: Checking Dataset...")
    local_txt_dir = os.path.join(BASE_DIR, "temp_txt_dataset")
    if not os.path.exists(local_txt_dir): os.makedirs(local_txt_dir)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]

    count = 0
    for path in files:
        fname = os.path.basename(path).replace(".pdf", ".txt").replace(" ", "_")
        txt_path = os.path.join(local_txt_dir, fname)

        if not os.path.exists(txt_path):
            try:
                reader = PdfReader(path)
                text = reader.pages[0].extract_text() or ""
                clean_text = text.encode("ascii", "ignore").decode("ascii")[:3000]
                full_prompt = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{clean_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_prompt)
                count += 1
            except:
                pass

    if count > 0:
        print(f"   ↳ Pushing {count} new files...")
        subprocess.run([ADB_PATH, "shell", f"mkdir -p {DEVICE_DATASET_DIR}"], check=True)
        subprocess.run([ADB_PATH, "push", local_txt_dir + "/.", DEVICE_DATASET_DIR],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ Upload Complete.")
    else:
        print("   ✅ Dataset ready.")


def parse_metrics(logs):
    gen_match = re.search(r'Generation:\s+(\d+\.\d+)\s+t/s', logs)
    if gen_match: return float(gen_match.group(1))
    ms_match = re.search(r'eval time =.*=\s+(\d+\.\d+)\s+ms per token', logs)
    if ms_match: return 1000.0 / float(ms_match.group(1))
    return 0.0


def clean_json(raw_output):
    if not raw_output: return None
    text = raw_output.replace(" || ", "\n")
    matches = re.finditer(r'\{.*?\}', text, re.DOTALL)
    candidates = []
    for m in matches:
        try:
            candidates.append(json.loads(m.group(0)))
        except:
            continue
    return candidates[-1] if candidates else None


def run_warm_benchmark():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    setup_device_dataset()

    processed_files = set()
    results = []

    if os.path.exists(CSV_PATH):
        try:
            processed_files = set(pd.read_csv(CSV_PATH)['filename'].tolist())
            print(f"🔄 RESUME: Skipping {len(processed_files)} processed files.")
        except:
            pass

    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]

    print("=" * 70)
    print(f"🔥 PNEUMA FINAL WARM RUN: {MODEL_NAME}")
    print("=" * 70 + "\n")

    for i, path in enumerate(pdf_files):
        fname_pdf = os.path.basename(path)
        fname_txt = fname_pdf.replace(".pdf", ".txt").replace(" ", "_")

        if fname_pdf in processed_files: continue

        print(f"--- [{i + 1}/{len(pdf_files)}] {fname_pdf} ---")

        try:
            device_file = f"{DEVICE_DATASET_DIR}/{fname_txt}"

            cmd = (
                f'{ADB_PATH} shell "cd {DEVICE_DIR} && '
                f'LD_LIBRARY_PATH={DEVICE_DIR}:/vendor/lib64:/system/lib64 '
                f'./pneuma_engine -m {DEVICE_MODEL_PATH} '
                f'-f {device_file} '
                f'-n 512 -c 2048 -ngl 99 -t 8 --temp 0 --simple-io < /dev/null"'
            )

            start_time = time.time()
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace'
            )

            full_log = []

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None: break

                if line:
                    clean_line = line.strip()
                    full_log.append(clean_line)

                    if "t/s" in clean_line and "Generation" in clean_line:
                        print(f"      [STATS] {clean_line}")
                        # 🔥 KILL SWITCH: We have the data, cut the cord.
                        process.kill()
                        break

                if time.time() - start_time > 120:
                    process.kill()
                    print("   ❌ TIMEOUT")
                    break

            output_str = "\n".join(full_log)
            tps = parse_metrics(output_str)
            data = clean_json(output_str)

            results.append({
                "filename": fname_pdf,
                "title": data.get("title", "N/A") if data else "N/A",
                "authors": str(data.get("authors", "N/A")) if data else "N/A",
                "doi": data.get("doi", "N/A") if data else "N/A",
                "tps": tps,
                "model": MODEL_NAME,
                "platform": PLATFORM_NAME,
                "raw_output": output_str.replace('\n', ' || ')
            })

            pd.DataFrame(results).to_csv(CSV_PATH, index=False)

            if tps > 0:
                print(f"   🚀 Saved. (TPS: {tps:.2f})")
            else:
                print("   ⚠️  No TPS captured.")

                # GIVE ANDROID TIME TO CLEAN UP RAM
            time.sleep(3.0)  # <--- Increased from 0.5

        except Exception as e:
            print(f"   ❌ Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    run_warm_benchmark()