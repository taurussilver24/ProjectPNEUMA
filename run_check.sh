#!/bin/bash

# Point to standard Linux CUDA paths (Native on B200 Cloud Images)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "🚀 LAUNCHING SERVER PROBE..."
echo "   Look for: 'ggml_cuda_init: found 1 CUDA devices: NVIDIA B200'"

# Just test LFM for now to prove it works
./llama_dist/llama-server \
    -m models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf \
    -ngl 99 \
    -c 32768 \
    --port 8080 \
    --host 0.0.0.0