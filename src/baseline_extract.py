import os
import re
import time
import json
import pandas as pd
import spacy
import fitz  # PyMuPDF

nlp = spacy.load("en_core_web_sm")

DATASET_PATH = "../dataset"
RESULTS_FILE = "../logs/results_TRADITIONAL_BASELINE.csv"


def sota_regex_extract(text):
    start_time = time.time()

    # 1. DOI EXTRACTION (Standard Academic Pattern)
    doi_match = re.search(r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b', text, re.I)
    doi = doi_match.group(0) if doi_match else "None"

    # 2. arXiv ID EXTRACTION
    arxiv_match = re.search(r'arXiv:\d{4}\.\d{4,5}', text)
    arxiv = arxiv_match.group(0) if arxiv_match else "None"

    # 3. HEURISTIC TITLE EXTRACTION
    # Rules: Skip headers like "arXiv:...", pick the first line with > 4 words that isn't a date
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 10]
    title = "Not Found"
    for line in lines[:10]:  # Look at first 10 lines
        if not re.search(r'(arXiv|downloaded|journal|conference|copyright|http|©)', line, re.I):
            if len(line.split()) >= 4:
                title = line
                break

    # 4. AUTHOR EXTRACTION (NER + Regex Fallback)
    # Using SpaCy to find Person entities in the first 1000 chars
    doc = nlp(text[:1500])
    authors = list(set([ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]))
    # Clean up authors (remove common false positives)
    authors = [a for a in authors if len(a.split()) >= 2 and len(a) < 40][:4]

    # 5. KEYWORD EXTRACTION
    keywords = []
    kw_block = re.search(r'(?i)keywords[:\s]+(.*?)(?:\n\n|\r\n\r\n|Abstract|Introduction)', text, re.S)
    if kw_block:
        keywords = [k.strip() for k in re.split(r'[,;]', kw_block.group(1)) if len(k) > 2][:5]

    end_time = time.time()

    return {
        "title": title,
        "authors": ", ".join(authors),
        "doi": doi,
        "arxiv_id": arxiv,
        "keywords": ", ".join(keywords),
        "processing_time_ms": (end_time - start_time) * 1000
    }


def run_baseline():
    results = []
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        return

    files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.pdf')]
    print(f"🚀 Running SOTA Regex Baseline on {len(files)} files...")

    start_bench = time.time()
    for file in files:
        try:
            doc = fitz.open(os.path.join(DATASET_PATH, file))
            text = doc[0].get_text()  # Get text from first page

            data = sota_regex_extract(text)
            data['filename'] = file
            results.append(data)
            doc.close()
        except Exception as e:
            print(f"Skipping {file}: {e}")

    total_time = time.time() - start_bench
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)

    print("-" * 40)
    print(f"COMPLETED in {total_time:.2f} seconds")
    print(f"Avg Speed: {len(files) / total_time:.2f} docs/sec")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_baseline()