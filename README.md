# Project PNEUMA: Private Intelligence Benchmark

**A Heterogeneous Hardware Analysis of SLM-based Metadata Extraction**

Project PNEUMA (Private Neural Metadata Analysis) evaluates the "Private Intelligence" paradigm—running local, air-gapped metadata extraction tasks on diverse hardware tiers using Small Language Models (SLMs).

---

## Project Structure
```
ProjectPNEUMA/
├── dataset/              # [GITIGNORE] Stores the 1,000 PDFs
├── logs/                 # Stores benchmark CSVs and Clash Reports
├── src/                  
│   ├── download.py       # Multi-model harvester (External SSD support)
│   ├── pneuma_bench.py   # The Master Benchmarker (Linux/Mac aware)
│   └── clash_inspector.py # Semantic Accuracy Evaluator
├── .gitignore            # Excludes massive .pdf and .gguf files
└── README.md             
```

---

## 1. Installation

### Hardware-Specific Engine

You must install `llama-cpp-python` with the correct acceleration for your current machine.

**For AMD RX 6800 (Linux/Vulkan):**
```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --upgrade --no-cache-dir
```

**For Apple M1 (macOS Metal):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --no-cache-dir
```

**For NVIDIA RTX 4080 (CUDA):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --no-cache-dir
```

---

## 2. Setup (The External SSD Workflow)

To support the 20GB+ model library across platforms, all `.gguf` files should be stored on an external SSD (e.g., `/run/media/taurus/Games/models/` or `/Volumes/Games/models/`).

1. **Harvest Models:** Update the `LOCAL_DIR` in `src/download.py` to point to your SSD mount, then run:
```bash
   python src/download.py
```
   Downloads: Qwen-2.5-3B, Phi-4 (14B-quant), and LFM-2.5 (Liquid).

2. **Sync Data:** Ensure your 15-file pilot (or 1,000-file marathon) `dataset/` is available locally in the project folder.

---

## 3. The "Clash" Benchmarking Protocol

The benchmark is designed to compare **Traditional Regex** against **Semantic SLMs** across three distinct architectures (Transformer vs. Liquid).

### Step A: The Pilot Test (15 Files)

Run the alphabetical pilot to ensure the hardware backend (Vulkan/Metal) is stable:

1. Edit `src/pneuma_bench.py` and set `FILE_LIMIT = 15`.
2. Set `MODE = "REGEX"` and run.
3. Set `MODE = "SLM"` and run (this will cycle through all 3 models).

### Step B: The Clash Inspection

Generate the Accuracy vs. Discovery report:
```bash
python src/clash_inspector.py
```

---

## 4. Hardware Comparison Matrix (Research Tiers)

| Tier | Unit | Memory | Backend | Status |
|------|------|--------|---------|--------|
| **Desktop** | AMD RX 6800 | 16 GB GDDR6 | Vulkan | ✅ Pilot Complete |
| **Unified** | Apple M1 | 16 GB Unified | Metal | 🔄 Next Priority |
| **Pro** | RTX 4080m | 12 GB GDDR6 | CUDA | 📅 Scheduled |

---

## Troubleshooting

- **"LFM-2.5 Not Found":** Verify the hyphen in the filename matches between `download.py` and `pneuma_bench.py`.
- **`llama_decode returned -1`:** Known instability on Vulkan for Liquid architectures. This is a **primary research finding** for the "Architectural Fragility" section of the paper.
- **Venv Path Errors:** If moving between machines, do **not** copy the `.venv`. Re-initialize it locally on the new machine.
- **GPU Not Detected:** Verify you installed `llama-cpp-python` with the correct `CMAKE_ARGS` for your hardware.
- **Out of Memory:** Reduce `n_ctx` in `pneuma_bench.py` or use smaller quantizations (Q4 instead of Q8).

---

## Research Context

This benchmark directly addresses **DJLIT's Special Issue** focus on "Responsible and Ethical Use of AI" by demonstrating:

1. **Privacy-Preserving Infrastructure:** All inference occurs locally without cloud dependencies
2. **Hardware Accessibility:** Performance characterization across consumer, prosumer, and professional tiers
3. **Architectural Diversity:** Comparative analysis of Transformer vs. Liquid Foundation Models
4. **Quantization Trade-offs:** INT4 vs. INT8 accuracy/performance analysis

---

## Citation

If you use this benchmark in your research, please cite:
```
[Your Name]. (2026). Scaling Private Intelligence: A Cross-Platform Performance 
Analysis of Quantized Small Language Models for Secure Metadata Extraction in 
Technical Repositories. DJLIT Special Issue on AI in Libraries.
```

---

## License

This project is released under the MIT License for academic and research purposes.