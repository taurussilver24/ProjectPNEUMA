import os
import sys

# --- WINDOWS DLL INJECTION ---
# You proved these files exist. Now we force Windows to use them.
try:
    base_path = os.path.join(sys.prefix, "Lib", "site-packages", "llama_cpp", "lib")
    os.add_dll_directory(base_path)
    print(f"🔧 DLL Path Added: {base_path}")
except Exception as e:
    print(f"⚠️ Warning: {e}")

from llama_cpp import Llama

# --- PATHS ---
# Adjust this to point to your .gguf file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")

if not os.path.exists(MODEL_PATH):
    print(f"❌ MODEL MISSING at: {MODEL_PATH}")
    sys.exit(1)

print("\n🏎️  ATTEMPTING ENGINE START...")
print("    Watch the logs below for 'BLAS = 1' or 'CUDA = 1'")

try:
    # This is the real test. If this crashes, we panic. If it runs, we win.
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # <--- FORCE GPU
        verbose=True  # <--- SHOW ME THE LOGS
    )
    print("\n✅ SUCCESS: ENGINE STARTED.")
    print("   If you see 'ggml_cuda_init: found 1 CUDA devices' above, you are GOLDEN.")

except Exception as e:
    print(f"\n❌ CRASH: {e}")