from huggingface_hub import hf_hub_download
import os

# --- PATH CONFIG ---
# Points to project root 'models' folder when run from 'src'
LOCAL_DIR = "../models/"

MODELS = [
    {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file": "qwen2.5-3b-instruct-q4_k_m.gguf"
    },
    {
        "repo": "itlwas/phi-4-Q4_K_M-GGUF",
        "file": "phi-4-q4_k_m.gguf"
    },
    {
        # CORRECTED REPO ID (Added hyphen to match official HF)
        "repo": "LiquidAI/LFM-2.5-1.2B-GGUF",
        "file": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    }
]

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"--- 🚀 HARVESTING MODELS TO CLOUD SSD ---")

for m in MODELS:
    print(f"\n📥 Target: {m['file']}")
    try:
        hf_hub_download(
            repo_id=m['repo'],
            filename=m['file'],
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✅ SUCCESS")
    except Exception as e:
        print(f"❌ ERROR: {e}")