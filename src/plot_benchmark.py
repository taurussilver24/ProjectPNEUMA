import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
from Levenshtein import ratio

# --- CONFIG ---
LOG_DIR = "../logs"
OUTPUT_WARM = "benchmark_warm_full.png"
OUTPUT_COLD = "benchmark_cold_full.png"
BASELINE_DIR = LOG_DIR  # Where regex baselines live


# --- HELPERS ---
def load_baseline(platform):
    """Finds the regex baseline to compare titles against."""
    plat_clean = platform.replace(" ", "_").replace("Apple_M1", "Apple_M1(Metal)")
    # Try specific, then fallback
    files = glob.glob(os.path.join(BASELINE_DIR, f"clash_REGEX_BASELINE_*{platform}*.csv"))
    if not files:
        files = glob.glob(os.path.join(BASELINE_DIR, "clash_REGEX_BASELINE_Apple_M1(Metal).csv"))

    if files:
        return pd.read_csv(files[0])
    return None


def is_verified_doi(doi_text):
    """Strict DOI check."""
    if not doi_text or pd.isna(doi_text): return False
    val = str(doi_text).lower().strip()
    if any(x in val for x in ["nnnn", "xxxx", "example", "not found"]): return False
    if re.match(r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$', val): return True
    return False


def parse_filename(filename):
    name = os.path.basename(filename).replace("clash_", "").replace(".csv", "")

    if "cold" in name.lower():
        mode = "COLD START"
    else:
        mode = "WARM START"

    if "Dimensity" in name:
        platform = "Dimensity 9300+"
    elif "Apple_M1" in name:
        platform = "Apple M1"
    elif "AMD" in name:
        platform = "AMD RX 6800"
    else:
        platform = "Unknown"

    if "LFM" in name:
        model = "LFM-2.5"
    elif "Phi" in name:
        model = "Phi-4"
    elif "Qwen" in name:
        model = "Qwen2.5"
    else:
        model = "Unknown"

    return model, mode, platform


def load_data():
    files = glob.glob(os.path.join(LOG_DIR, "clash_*.csv"))
    files = [f for f in files if "REGEX" not in f and "final" not in f]

    data = []

    # Pre-load baselines to avoid re-reading
    baselines = {}

    for f in files:
        model, mode, platform = parse_filename(f)

        # Load baseline for this platform if not cached
        if platform not in baselines:
            baselines[platform] = load_baseline(platform)

        try:
            df = pd.read_csv(f)
            df = df[df['title'] != 'ERROR']

            # 1. SPEED
            tps = df['tps'].mean()

            # 2. DOI ACCURACY (Precision)
            if 'doi' in df.columns:
                valid = df['doi'].apply(is_verified_doi).sum()
                doi_acc = (valid / len(df)) * 100
            else:
                doi_acc = 0.0

            # 3. TITLE SIMILARITY (Recall/Context)
            title_sim = 0.0
            if baselines[platform] is not None:
                # Merge on filename to compare correct rows
                m = pd.merge(df, baselines[platform], on="filename", suffixes=("_slm", "_reg"))
                if not m.empty:
                    # Levenshtein Ratio (0-100)
                    sims = m.apply(lambda r: ratio(str(r.title_slm), str(r.title_reg)), axis=1)
                    title_sim = sims.mean() * 100

            if tps > 0 and platform != "Unknown":
                data.append({
                    "Platform": platform,
                    "Model": model,
                    "Mode": mode,
                    "Speed (TPS)": tps,
                    "DOI Accuracy (%)": doi_acc,
                    "Title Similarity (%)": title_sim
                })
        except:
            pass

    return pd.DataFrame(data)


def add_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')


def generate_mode_plot(df, mode, output_file):
    subset = df[df["Mode"] == mode]
    if subset.empty: return

    sns.set_theme(style="whitegrid")
    # 1 Row, 3 Columns (Speed | DOI | Title)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- PLOT 1: SPEED ---
    order = subset.sort_values("Speed (TPS)", ascending=False)["Platform"].unique()
    sns.barplot(data=subset, x="Platform", y="Speed (TPS)", hue="Model",
                ax=axes[0], palette="viridis", alpha=0.9, order=order)
    axes[0].set_title("Speed (Tokens/Sec)", fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    add_labels(axes[0])

    # --- PLOT 2: DOI ACCURACY (Hard Metric) ---
    sns.barplot(data=subset, x="Platform", y="DOI Accuracy (%)", hue="Model",
                ax=axes[1], palette="magma", alpha=0.9, order=order)
    axes[1].set_title("Precision (Valid DOI %)", fontsize=14, fontweight='bold')
    axes[1].get_legend().remove()
    axes[1].set_ylim(0, 50)  # Scale for DOI
    add_labels(axes[1])

    # --- PLOT 3: TITLE SIMILARITY (Soft Metric) ---
    sns.barplot(data=subset, x="Platform", y="Title Similarity (%)", hue="Model",
                ax=axes[2], palette="coolwarm", alpha=0.9, order=order)
    axes[2].set_title("Context (Title Similarity %)", fontsize=14, fontweight='bold')
    axes[2].get_legend().remove()
    axes[2].set_ylim(0, 110)  # Scale for Title (can be high)
    add_labels(axes[2])

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Saved 3-metric benchmark to: {output_file}")


if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        generate_mode_plot(df, "WARM START", OUTPUT_WARM)
        generate_mode_plot(df, "COLD START", OUTPUT_COLD)