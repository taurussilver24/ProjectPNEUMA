# Project PNEUMA: Scaling Private Intelligence

### A Cross-Platform Analysis of Stateless vs. Stateful SLM Inference

**Authors:** Rishit Arora (Kobe Institute of Computing, Osaka Metropolitan University)

Project **PNEUMA** (Private Neural Metadata Analysis) validates the **"Private Intelligence"** paradigm—executing secure, air-gapped metadata extraction on heterogeneous hardware. This repository contains the benchmarking tools used to discover the **"Stateless Paradox,"** where re-initializing Liquid Foundation Models (LFMs) for every document outperforms traditional stateful Transformer inference in both speed and stability.

> **🚨 BREAKTHROUGH (Jan 2026):** Mobile Edge hardware (Dimensity 9300+) has successfully outperformed Desktop-class Silicon (Apple M1) in sustained inference speed (31.5 TPS vs 27.0 TPS) using the **PneumaEdge** bridge.

---

## 📂 Project Structure

```
ProjectPNEUMA/
├── dataset/                    # [GITIGNORE] The 1,000 PDF Corpus
├── logs/                       # Benchmark CSVs (Raw) and Clash Reports (Verified)
├── models/                     # Symlinks to external SSD model storage
├── src/
│   ├── pneuma_bench.py         # Main Benchmark (PC/Mac)
│   ├── pneuma_android_bridge.py# [NEW] ADB Bridge for Mobile Edge Benchmark
│   ├── pneuma_extract.py       # Warm Benchmark for Transformers
│   ├── lfm_bench_cold.py       # Cold Benchmark for LFM
│   ├── clash_inspector.py      # The "Hallucination Shield" & Verified Accuracy logic
│   ├── regenerate_baseline.py  # Generates the Regex Ground Truth
│   └── clean_results.py        # Post-processing for raw JSON logs
├── .gitignore
└── README.md
```

---

## 1. 📦 Dataset Setup

The benchmark relies on a curated corpus of 1,000 technical PDFs (arXiv `cs.AI`).

### Option A: Manual Download (Google Drive)

1. Download the corpus: **[LINK TO DATASET]**
2. Place the zip file in the root directory.
3. Unzip to the `dataset/` folder:

```bash
mkdir -p dataset
unzip pneuma_1k_v1.zip -d dataset/
```

### Option B: Automated Harvest

If you prefer to build the dataset from scratch:

```bash
python src/pneuma_fetch.py --limit 1000 --category cs.AI
```

---

## 2. ⚙️ Installation & Acceleration

To replicate the "Stateless Paradox" results, you must install `llama-cpp-python` with the correct hardware backend.

### Tier 1: Desktop (AMD RX 6800 - Vulkan)
*The primary discovery platform for the Liquid LFM Paradox.*

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --upgrade --no-cache-dir --force-reinstall
```

### Tier 2: Unified Memory (Apple M1 - Metal)
*Used for comparing ARM-based Desktop vs. ARM-based Mobile performance.*

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --no-cache-dir --force-reinstall
```

### Tier 3: Edge Mobile (Dimensity 9300+ - Android)
*The Champion Tier.*

This tier requires the **PneumaEdge** engine (a custom Android C++ build of `llama.cpp` optimized for Immortalis GPUs).

1.  **Clone the Engine:** Go to the [PneumaEdge Repository](https://github.com/taurussilver24/PNEUMAEdge) to build the `pneuma_engine` binary.
2.  **Bridge Connection:** Use the bridge script in *this* repo to orchestrate the benchmark over ADB.

```bash
# Verify ADB Connection
adb devices

# Run the Bridge (Orchestrates the phone to process files)
python src/pneuma_android_bridge.py
```

---

## 3. 🚀 Benchmarking Protocol

The research follows a strict **"Clash"** methodology to compare Semantic Recall vs. Regex Precision.

### Step A: The Baseline

Generate the Regex "Ground Truth" to compare SLMs against.

```bash
python src/regenerate_baseline.py
```

### Step B: The Inference Marathon

Run the benchmark. The script automatically detects your hardware backend.

* **Warm Mode:** Keeps model in VRAM (Standard Transformer protocol).
* **Cold Mode:** Re-initializes model per file (The **LFM/Phi-4** optimal protocol).

```bash
# Run 1,000 files in Cold Start mode (Stateless)
python src/pneuma_bench.py --mode COLD --files 1000
```

### Step C: The Clash Inspection (Verified Accuracy)

Run the inspector to filter hallucinations (e.g., `10.1234/example.doi`). This generates the **Verified DOI** metrics for the paper.

```bash
python src/clash_inspector.py
```

---

## 4. 📊 Hardware Comparison Matrix

| Tier | Hardware | Backend | Status | Key Finding |
|------|----------|---------|--------|-------------|
| **Desktop** | AMD RX 6800 | Vulkan | ✅ Complete | **LFM Paradox:** Cold (56 TPS) > Warm (52 TPS) |
| **Unified** | Apple M1 | Metal | ✅ Complete | **Throttling:** Peak 27 TPS, Sustained 25 TPS |
| **Edge** | Dimensity 9300+ | PneumaEdge | 🏆 **Winner** | **Mobile Supremacy:** Sustained **31.5 TPS** (Beats M1) |
| **Pro** | RTX 4080 Mobile | CUDA | 📅 Feb | Control Group |

---

## 5. 🔬 Research Context

This repository accompanies the Special Issue submission to **DJLIT**.

**Key Definitions:**

* **Stateless Paradox:** The phenomenon where 3B-class models (LFM, Phi-4) achieve higher throughput and stability when re-loaded for every document, inverting traditional "Batch Processing" wisdom.
* **Verified DOI:** A metric that excludes model hallucinations (e.g., placeholders) to measure true semantic utility.
* **PneumaEdge:** The custom Android runtime environment (separate repo) that enables unprivileged, root-free execution of quantized LLMs via ADB.

---

## ⚖️ License & Citation

**License:** MIT License.

**Citation:**

> Arora, R. (2026). *Scaling Private Intelligence: A Cross-Platform Performance Analysis of Quantized Small Language Models for Secure Metadata Extraction*. Osaka Metropolitan University / Kobe Institute of Computing.