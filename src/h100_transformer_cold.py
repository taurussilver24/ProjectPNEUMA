import time
import json
import os
import glob
import re
import gc
import pandas as pd
from pypdf import PdfReader
from llama_cpp import Llama

# --- CONFIGURATION ---
PLATFORM_NAME = "NVIDIA H100 SXM (CUDA)"
FILE_LIMIT = 1000  # Full Dataset

# --- MULTI-MODEL QUEUE ---
# Matches filenames in setup_h100.sh
MODEL_QUEUE = [
    {"name": "Qwen2.5-3B", "file": "qwen3b.gguf"},
    {"name": "Phi-4", "file": "phi4.gguf"}
]

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "results")

# --- SYSTEM PROMPT: CHAIN OF THOUGHT ---
# We use string variables for backticks to prevent UI formatting issues
JSON_BLOCK_START = "```json"
JSON_BLOCK_END = "```"

SYSTEM_PROMPT = f"""You are an expert research librarian. Analyze the document to extract metadata.

STEP 1: REASONING
First, think silently about the document structure:
- Identify the main title (usually largest font/first page).
- Distinguish the actual authors from affiliations.
- Locate the specific DOI of *this* paper, ignoring DOIs in the references section.
- Synthesize the core contribution for the summary.

STEP 2: EXTRACTION
After your reasoning, output the final metadata in this exact JSON format inside a code block:

{JSON_BLOCK_START}
{{
    "title": "Exact string",
    "authors": ["List of strings"],
    "doi": "String or null",
    "arxiv_id": "String or null",
    "keywords": ["List of strings"],
    "summary": "String"
}}
{JSON_BLOCK_END}
"""


def extract_json_from_cot(text):
    """
    Robustly extracts JSON block from Chain-of-Thought markdown output.
    """
    try:
        # 1. Try finding code block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # 2. Try finding raw JSON object if code block missing
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return {}
    except:
        return {}


def run_transformer_cold():
    # Validation
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Error: Model dir not found at {MODEL_DIR}")
        return

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))[:FILE_LIMIT]

    # --- MODEL LOOP ---
    for model_config in MODEL_QUEUE:
        model_name = model_config["name"]
        model_file = model_config["file"]
        model_path = os.path.join(MODEL_DIR, model_file)

        if not os.path.exists(model_path):
            print(f"⚠️ Skipping {model_name}: File not found at {model_path}")
            continue

        print("\n" + "=" * 70)
        print(f"❄️ {model_name.upper()} COLD START (CoT) | H100 BENCHMARK")
        print(f"Platform: {PLATFORM_NAME}")
        print(f"Target:   {len(files)} Files")
        print("=" * 70 + "\n")

        results = []

        # --- FILE LOOP ---
        for i, path in enumerate(files):
            fname = os.path.basename(path)
            llm = None

            try:
                reader = PdfReader(path)
                if len(reader.pages) > 0:
                    text = reader.pages[0].extract_text()[:3500]
                else:
                    text = ""

                # 1. COLD START INIT (Reload for every file)
                start_total = time.time()

                llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,  # Full Offload
                    main_gpu=0,
                    n_ctx=8192,  # 8k Context for CoT
                    n_batch=512,
                    verbose=False
                )

                # 2. INFERENCE (Reasoning + JSON)
                start_infer = time.time()
                resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    # CRITICAL: NO 'json_object' mode here. It breaks CoT.
                    temperature=0.1,
                    max_tokens=1024
                )

                end_time = time.time()
                inference_time = end_time - start_infer
                total_time = end_time - start_total  # Load + Gen

                content = resp['choices'][0]['message']['content']
                tokens = resp['usage']['completion_tokens']
                tps = tokens / inference_time if inference_time > 0 else 0

                # Extract JSON from the reasoning text
                data = extract_json_from_cot(content)

                results.append({
                    "filename": fname,
                    "title": data.get("title", "N/A"),
                    "authors": str(data.get("authors", "N/A")),
                    "doi": data.get("doi", "None"),
                    "arxiv_id": data.get("arxiv_id", "None"),
                    "keywords": str(data.get("keywords", "None")),
                    "summary": data.get("summary", "N/A"),
                    "tps": round(tps, 2),
                    "inference_time": round(inference_time, 3),
                    "total_time": round(total_time, 3),
                    "tokens": tokens,
                    "model": model_name,
                    "mode": "cold_cot",
                    "platform": PLATFORM_NAME
                })

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"[{i + 1}/{len(files)}] {fname} | TPS: {tps:.2f} | Load+Gen: {total_time:.2f}s")

            except Exception as e:
                print(f"❌ Error on {fname}: {e}")

            finally:
                # 3. MEMORY PURGE (The Cold Reset)
                if llm:
                    del llm
                gc.collect()

        # 4. SAVE RESULTS
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
        output_csv = os.path.join(LOG_DIR, f"h100_{model_name.lower()}_cot_cold.csv")
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"\n✅ Saved {model_name} results to: {output_csv}")

        # Extra pause between models
        time.sleep(5)


if __name__ == "__main__":
    run_transformer_cold()