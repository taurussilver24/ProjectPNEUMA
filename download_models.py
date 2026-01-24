from huggingface_hub import hf_hub_download
import os
import shutil

LOCAL_DIR = "models/"
os.makedirs(LOCAL_DIR, exist_ok=True)

# CONFIG: Repo -> Source File -> Local Destination Name
TARGETS = [
    {
        "repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "src": "LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
        "dest": "lfm.gguf"
    }
]

print(f"--- 🚀 DOWNLOADING MODELS ---")

for t in TARGETS:
    print(f"\n📥 Fetching: {t['repo']}...")
    try:
        # Download to cache/local dir
        file_path = hf_hub_download(
            repo_id=t['repo'],
            filename=t['src'],
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )

        # Rename to the standardized name expected by benchmark scripts
        final_path = os.path.join(LOCAL_DIR, t['dest'])
        os.rename(file_path, final_path)

        print(f"✅ Ready: {t['dest']}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)
