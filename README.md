# Project PNEUMA: Scaling Private Intelligence

### A Cross-Platform Analysis of Stateless vs. Stateful SLM Inference

**Authors:** Rishit Arora(Kobe Institute of Computing, Osaka Metropolitan University)

Project **PNEUMA** (Private Neural Metadata Analysis) validates the **"Private Intelligence"** paradigm—executing secure, air-gapped metadata extraction on heterogeneous hardware. This repository contains the benchmarking tools used to discover the **"Stateless Paradox,"** where re-initializing Liquid Foundation Models (LFMs) for every document outperforms traditional stateful Transformer inference in both speed and stability.

---

## 📂 Project Structure

```
ProjectPNEUMA/
├── dataset/                    # [GITIGNORE] The 1,000 PDF Corpus
├── logs/                       # Benchmark CSVs (Raw) and Clash Reports (Verified)
├── models/                     # Symlinks to external SSD model storage
├── src/
│   ├── pneuma_bench.py         # Test Benchmark
│   ├── pneuma_extract.py       # Warm Benchmark for Transformers
│   ├── pneuma_fetch.py         # Dataset is already provided but in case you want your own dataset
│   ├── lfm_warm_diagnostic.py  # Warm Benchmark for LFM
│   ├── lfm_bench_cold.py       # Cold Benchmark for LFM
│   ├── transformer_bench_cold  # Cold Benchmark for Transformers
│   ├── clash_inspector.py      # The "Hallucination Shield" & Verified Accuracy logic
│   ├── regenerate_baseline.py  # Generates the Regex Ground Truth
│   └── download.py             # Model Harvester (HuggingFace)
├── .gitignore
└── README.md
```

---

## 1. 📦 Dataset Setup

The benchmark relies on a curated corpus of 1,000 technical PDFs (arXiv `cs.AI`).

### Option A: Manual Download (Google Drive)

1. Download the corpus: **https://drive.google.com/file/d/1_jZObj5C1A9k-Q5u27ntGhSX2FD_VeJc/view?usp=drive_link**
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
*Currently benchmarking for "Template Fatigue" analysis.*

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --no-cache-dir --force-reinstall
```

### Tier 3: Edge Mobile (Dimensity 9300+ - Android)
*Experimental Tier: Requires MLC-LLM compilation (See `docs/android_build.md`).*

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
| **Desktop** | AMD RX 6800 | Vulkan | ✅ Complete | **LFM Paradox:** Cold ( TPS) > Warm ( TPS) |
| **Unified** | Apple M1 | Metal | 🔄 Active | **Template Fatigue:** Phi-4 collapses to 2% recall |
| **Edge** | Dimensity 9300 | MLC/Vulkan | ⏳ Pending | Building `libtvm4j` bindings |
| **Pro** | RTX 4080 Mobile | CUDA | 📅 Feb | Control Group |

---

## 5. 🔬 Research Context

This repository accompanies the Special Issue submission to **DJLIT**.

**Key Definitions:**

* **Stateless Paradox:** The phenomenon where 3B-class models (LFM, Phi-4) achieve higher throughput and stability when re-loaded for every document, inverting traditional "Batch Processing" wisdom.
* **Verified DOI:** A metric that excludes model hallucinations (e.g., placeholders) to measure true semantic utility.
* **Template Fatigue:** A failure mode observed on low-power silicon (M1) where models default to LaTeX templates instead of extracting real data.

---

## ⚖️ License & Citation

**License:** MIT License.

**Citation:**

> Arora, R.(2026). *Scaling Private Intelligence: A Cross-Platform Performance Analysis of Quantized Small Language Models for Secure Metadata Extraction*. Osaka Metropolitan University / Kobe Institute of Computing.