import arxiv
import os
from tqdm import tqdm

# CONFIGURATION
TARGET_COUNT = 1005# Set to 1000 for the full run, or 10 for a test
DATA_DIR = "../dataset"
QUERY = "cat:cs.AI"  # Category: Computer Science > AI


def fetch_pdfs():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    client = arxiv.Client()
    search = arxiv.Search(
        query=QUERY,
        max_results=TARGET_COUNT,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    print(f"--- PNEUMA DATA COLLECTION: Target {TARGET_COUNT} PDFs ---")

    downloaded = 0
    results = client.results(search)

    for paper in tqdm(results, total=TARGET_COUNT, desc="Downloading"):
        if downloaded >= TARGET_COUNT:
            break

        # Clean filename to avoid OS errors
        safe_title = "".join([c for c in paper.title if c.isalnum() or c in " ._-"]).strip()
        safe_title = safe_title[:50]  # Truncate to avoid long paths
        filename = f"{paper.entry_id.split('/')[-1]}_{safe_title}.pdf"
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            try:
                paper.download_pdf(dirpath=DATA_DIR, filename=filename)
                downloaded += 1
            except Exception as e:
                print(f"Failed {filename}: {e}")
        else:
            downloaded += 1

    print(f"\nSUCCESS: Dataset ready at {DATA_DIR} ({downloaded} files)")


if __name__ == "__main__":
    fetch_pdfs()