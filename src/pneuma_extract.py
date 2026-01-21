import time
import json
import os
import glob
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- CROSS-PLATFORM CONFIGURATION ---
# Change this string based on where you are running!
# Options: "AMD RX 6800 (Vulkan)", "Apple M1 (Metal)", "NVIDIA H100 (CUDA)"
PLATFORM_NAME = "AMD RX 6800 (Vulkan)"

# Paths relative to the script execution or absolute
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "../models", "qwen2.5-3b-instruct-q4_k_m.gguf")
DATA_DIR = os.path.join(BASE_DIR, "../dataset")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Model Settings
GPU_LAYERS = 37  # -1 for "All Layers" (Recommended for M1/H100/RX6800)
CTX_SIZE = 4096  # Context window
MAX_TOKENS = 600  # Increased from 512 to avoid truncation JSON errors

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a specialized metadata extraction system. 
Analyze the provided document text and extract the following fields into a valid JSON object:
{
    "title": "Exact paper title",
    "authors": ["List of author names"],
    "keywords": ["List of keywords if present"],
    "summary": "A 1-sentence summary of the abstract"
}
Output ONLY the JSON. Do not add markdown formatting."""


def init_model():
    print(f"--- SYSTEM: Initializing {PLATFORM_NAME} ---")
    print(f"--- MODEL: {MODEL_PATH} ---")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please download it!")

    # llama-cpp-python auto-detects the backend (Vulkan/Metal/CUDA)
    # based on how it was installed.
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=CTX_SIZE,
        verbose=False
    )
    return llm


def extract_text_from_pdf(filepath):
    try:
        reader = PdfReader(filepath)
        text = reader.pages[0].extract_text()
        return text[:3500]
    except Exception:
        return None


def run_benchmark():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

    llm = init_model()
    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))

    results = []
    print(f"--- BENCHMARK START: {len(pdf_files)} Documents ---")

    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        raw_text = extract_text_from_pdf(pdf_path)

        if not raw_text:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"DOCUMENT TEXT:\n{raw_text}"}
        ]

        # --- TIMING ---
        start_time = time.time()

        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            end_time = time.time()
            total_time = end_time - start_time

            usage = response['usage']
            completion_tokens = usage['completion_tokens']
            prompt_tokens = usage['prompt_tokens']
            tps = completion_tokens / total_time

            output_content = response['choices'][0]['message']['content']

            # Validation
            try:
                json.loads(output_content)
                valid_json = True
            except:
                valid_json = False

            print(f"[{i + 1}/{len(pdf_files)}] {filename} | TPS: {tps:.2f} | JSON: {valid_json}")

            results.append({
                "filename": filename,
                "platform": PLATFORM_NAME,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_time_sec": total_time,
                "tokens_per_second": tps,
                "valid_json": valid_json
            })
        except Exception as e:
            print(f"Error on {filename}: {e}")

        # Save checkpoint every 50 files
        if i % 50 == 0:
            pd.DataFrame(results).to_csv(os.path.join(LOG_DIR, f"results_{PLATFORM_NAME.replace(' ', '_')}.csv"),
                                         index=False)

    # Final Save
    pd.DataFrame(results).to_csv(os.path.join(LOG_DIR, f"results_{PLATFORM_NAME.replace(' ', '_')}.csv"), index=False)
    print("--- BENCHMARK COMPLETE ---")


if __name__ == "__main__":
    run_benchmark()