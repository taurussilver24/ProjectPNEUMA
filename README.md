# Project PNEUMA: Private Intelligence Benchmark

A cross-platform framework for benchmarking Quantized Small Language Models (SLMs) on heterogeneous hardware.

## Project Structure
```
ProjectPNEUMA/
├── dataset/              # [GITIGNORE] Stores the 1,000 PDFs
├── models/               # [GITIGNORE] Stores Qwen2.5-3B.gguf
├── logs/                 # Stores benchmark CSVs (e.g., rx6800_results.csv)
├── src/                  # Source code
│   ├── pneuma_fetch.py   # The Crawler (optional - dataset pre-downloaded)
│   └── pneuma_extract.py # The Benchmarker
├── .gitignore            # Critical to avoid uploading 2GB of PDFs to GitHub
├── requirements.txt      # Common libraries
└── README.md             # The Manual for M1/H100 setup
```

## 1. Installation

### Common Requirements
```bash
pip install -r requirements.txt
```

### Hardware-Specific Engine
You must install `llama-cpp-python` with the correct hardware acceleration.

**For AMD RX 6800 (Linux/Vulkan):**
```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

**For Apple M1/M2/M3 (macOS Metal):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

**For NVIDIA H100/4080 (CUDA):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

## 2. Setup Data & Model

### Option A: Download Pre-Collected Dataset (Recommended)

1. **Download Dataset from Google Drive:**
```bash
   # Download dataset.zip from:
   # https://drive.google.com/file/d/YOUR_FILE_ID_HERE/view?usp=sharing
```

2. **Extract the Dataset:**
```bash
   unzip dataset.zip
   # This will create the dataset/ folder with 1,000 PDFs
```

3. **Download the Model:** Place `qwen2.5-3b-instruct-q4_k_m.gguf` inside the `models/` folder.
   - Download from: [Hugging Face - Qwen2.5-3B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)

### Option B: Collect Dataset Yourself (Optional)

If you want to collect your own dataset:
```bash
python3 src/pneuma_fetch.py
```
**Note:** This will download 1,000 random arXiv PDFs. For scientific consistency across hardware platforms, use Option A instead.

## 3. Run Benchmark

1. Open `src/pneuma_extract.py`.
2. Edit the `PLATFORM_NAME` variable (e.g., set to "Apple M1", "RX 6800", "H100").
3. Run the benchmark:
```bash
   python3 src/pneuma_extract.py
```

4. Results are saved to `logs/PLATFORM_NAME_results.csv`.

---

## Dataset Information

- **Size:** 1,000 arXiv PDFs (~2GB compressed, ~2.5GB uncompressed)
- **Source:** arXiv.org open-access repository
- **Fields:** Computer Science (cs.*) and related domains
- **Purpose:** Standardized benchmark corpus for cross-platform comparison

**Critical:** All hardware platforms (RX 6800, M1, H100) must use the **exact same dataset** to ensure valid performance comparisons.

---

## Quick Start Example
```bash
# 1. Download and extract dataset
wget "https://drive.google.com/file/d/1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc/view?usp=sharing" -O dataset.zip
unzip dataset.zip

# 2. Install llama-cpp-python (example for Vulkan/AMD)
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir

# 3. Run benchmark
python3 src/pneuma_extract.py
```

## Troubleshooting

- **GPU not detected:** Check that you installed llama-cpp-python with the correct CMAKE_ARGS for your hardware.
- **Out of memory:** Reduce `n_ctx` in `pneuma_extract.py` or use a smaller quantization (Q4 instead of Q8).
- **Dataset extraction fails:** Ensure you have ~3GB free disk space.