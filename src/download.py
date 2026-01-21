from huggingface_hub import hf_hub_download
import os

# CONFIGURATION
REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
LOCAL_DIR = "../models"

print(f"--- STARTING DOWNLOAD ---")
print(f"Repo: {REPO_ID}")
print(f"File: {FILENAME}")

# Ensure directory exists
os.makedirs(LOCAL_DIR, exist_ok=True)

# Download
file_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False  # Crucial for GGUF to work with llama.cpp
)

print(f"\nSUCCESS! Model saved to:\n{file_path}")