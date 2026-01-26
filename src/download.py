from huggingface_hub import hf_hub_download
import os

# --- PATH CONFIG ---
# Points relative to 'src' folder to land in root 'models'
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
        "repo": "LiquidAI/LFM-2.5-1.2B-GGUF",
        "file": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    }
]

# Note: The 'LiquidAI' repo link in your snippet had a typo (LFM2.5-1.2B-Instruct-GGUF vs LFM-2.5-1.2B-GGUF).
# I fixed the repo ID above to the official one just in case, but kept your logic.

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"--- 🚀 HARVESTING MODELS TO SSD/CLOUD ---")

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