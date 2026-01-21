import time
import os
from llama_cpp import Llama

# --- PNEUMA CONFIGURATION ---
MODEL_PATH = "../models/qwen2.5-3b-instruct-q4_k_m.gguf"
# -1 = Offload ALL layers (The "God Mode" setting)
GPU_LAYERS = -1
CTX_SIZE = 4096


def run_benchmark():
    print(f"--- SYSTEM: PNEUMA INITIALIZATION [RX 6800] ---")

    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
        return

    print(f"Target: {MODEL_PATH}")
    print("Initializing Model with ROCm Offloading...")

    # Load Model
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=GPU_LAYERS,
        n_ctx=CTX_SIZE,
        n_batch=512,  # Optimizes prompt processing
        verbose=True  # We need the logs to confirm GPU usage
    )

    # --- DIAGNOSTIC CHECK ---
    # This tells us if the Python variable was actually accepted
    print(f"\n[DIAGNOSTIC] Requested GPU Layers: {GPU_LAYERS}")
    # Note: We rely on the internal C++ logs to confirm the actual offload

    # The Prompt
    prompt = """
<|im_start|>system
You are a technical librarian. Extract the Title and Keywords from the text.<|im_end|>
<|im_start|>user
Text: "We present a novel approach to optimizing INT4 quantization on AMD RDNA2 architectures using autotuning kernels."<|im_end|>
<|im_start|>assistant
"""

    print("\n--- PHASE 1: WARMUP (Compiling Kernels) ---")
    llm(prompt, max_tokens=10)

    print("\n--- PHASE 2: INFERENCE BENCHMARK ---")
    start_time = time.time()

    output = llm(
        prompt,
        max_tokens=200,
        stop=["<|im_end|>"],
        echo=False
    )

    end_time = time.time()
    total_time = end_time - start_time

    # Metrics
    tokens_gen = output['usage']['completion_tokens']
    tps = tokens_gen / total_time

    print("\n" + "=" * 40)
    print(f"OUTPUT: {output['choices'][0]['text'].strip()}")
    print("=" * 40)
    print(f"Tokens Generated: {tokens_gen}")
    print(f"Total Time:       {total_time:.4f} s")
    print(f"Throughput:       {tps:.2f} t/s")
    print("=" * 40)


if __name__ == "__main__":
    run_benchmark()