import pandas as pd
import os
from Levenshtein import ratio

# --- 1. INSPECTION CONFIGURATION ---
PLATFORM_NAME = "AMD RX 6800 (Vulkan)"  # Must match benchmarker
MODELS_TO_COMPARE = ["Qwen2.5-3B", "Phi-4", "LFM-2.5"]
REG_FILE = f"../logs/clash_REGEX_BASELINE_{PLATFORM_NAME.replace(' ', '_')}.csv"


def calculate_discovery_rate(df, column):
    """Calculates % of rows where data is NOT 'None' or 'N/A'"""
    valid_count = df[column].apply(lambda x: str(x).lower() not in ['none', 'n/a', 'nan', '']).sum()
    return (valid_count / len(df)) * 100


def inspect():
    if not os.path.exists(REG_FILE):
        print(f"❌ Error: Regex baseline missing at {REG_FILE}")
        return

    df_reg = pd.read_csv(REG_FILE)
    summary_data = []

    print("\n" + "=" * 80)
    print(f"PNEUMA CLASH INSPECTOR | Platform: {PLATFORM_NAME}")
    print("=" * 80)

    for model_name in MODELS_TO_COMPARE:
        slm_file = f"../logs/clash_{model_name}_{PLATFORM_NAME.replace(' ', '_')}.csv"

        if not os.path.exists(slm_file):
            print(f"⚠️ Skipping {model_name}: Result file not found.")
            continue

        df_slm = pd.read_csv(slm_file)
        m = pd.merge(df_slm, df_reg, on="filename", suffixes=("_slm", "_reg"))

        # --- METRICS ---
        # 1. Title Similarity (Cleanliness check)
        title_sim = m.apply(lambda r: ratio(str(r.title_slm), str(r.title_reg)), axis=1).mean()

        # 2. DOI & arXiv Discovery (The "Intelligence" delta)
        doi_rate_slm = calculate_discovery_rate(df_slm, 'doi')
        doi_rate_reg = calculate_discovery_rate(df_reg, 'doi')

        # 3. Keyword Yield (Semantic tagging)
        kw_rate_slm = calculate_discovery_rate(df_slm, 'keywords')

        # 4. Author Noise Reduction (Chars removed)
        # Note: If negative, it means SLM found MORE authors than Regex (Higher Recall)
        noise_reduction = df_reg['authors'].str.len().mean() - df_slm['authors'].str.len().mean()

        summary_data.append({
            "Model": model_name,
            "Avg_TPS": round(df_slm['tps'].mean(), 2),
            "Title_Sim": f"{title_sim:.2f}",
            "DOI_Discovery": f"{doi_rate_slm:.1f}%",
            "KW_Yield": f"{kw_rate_slm:.1f}%",
            "Author_Delta": round(noise_reduction, 1)
        })

    # --- FINAL REPORT ---
    report_df = pd.DataFrame(summary_data)
    print(report_df.to_string(index=False))

    print("\n" + "-" * 80)
    print(f"BASELINE (Regex) DOI Discovery: {calculate_discovery_rate(df_reg, 'doi'):.1f}%")
    print(f"Note: Negative Author_Delta suggests higher SLM author recall/discovery.")
    print("-" * 80)


if __name__ == "__main__":
    inspect()