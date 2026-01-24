import pandas as pd
import os
import re
from Levenshtein import ratio

# --- 1. INSPECTION CONFIGURATION ---
# Toggle this between "AMD_RX_6800_(Vulkan)" and "Apple_M1(Metal)" Dimensity_9300+(Vulkan)
PLATFORM_NAME = "Dimensity_9300+(Vulkan)"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Model configurations: defined names and the mode strings to look for
MODELS_TO_COMPARE = [
    {"name": "Qwen2.5-3B", "modes": ["warm", "cold_start"]},
    {"name": "Phi-4", "modes": ["warm", "cold_start"]},
    # LFM often uses "cold" in filename instead of "cold_start"
    {"name": "LFM-2.5", "modes": ["warm", "cold"]}
]

REG_FILE = os.path.join(LOG_DIR, f"clash_REGEX_BASELINE_{PLATFORM_NAME.replace(' ', '_')}.csv")

# --- 2. INTEGRITY FILTERS (The "Hallucination Shield") ---
DOI_BLACKLIST = [
    "nnnnn", "xxxx", "example", "1234", "not found", "not provided",
    "to be determined", "futuredoi", "doi_string", "yoursite", "abcd123",
    "insert doi", "doi:", "pmlr"
]


def is_verified_doi(doi_text):
    """
    Rigorous check for real DOIs vs. SLM hallucinations/templates.
    Returns True only if it looks like a valid, non-template DOI.
    """
    if not doi_text or pd.isna(doi_text): return False
    val = str(doi_text).lower().strip()

    # 1. Check Blacklist (Placeholders)
    if any(p in val for p in DOI_BLACKLIST): return False

    # 2. Check for repeating template characters (e.g., nnnnnnn or xxxxxxx)
    if re.search(r'[nx]{3,}', val): return False

    # 3. Standard DOI Regex: 10.prefix/suffix OR full URL validation
    # Matches: 10.1145/3786583.3786898 or 10.1016/j.ai.2024.03.012
    doi_pattern = r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$'

    if re.match(doi_pattern, val):
        return True

    if "doi.org/10." in val:
        # Check if the URL actually ends in a template pattern
        if "xxxx" in val or "nnnn" in val: return False
        return True

    return False


def calculate_verified_rate(df, column):
    """Calculates % of rows passing the integrity filter."""
    if column == 'doi':
        valid_count = df[column].apply(is_verified_doi).sum()
    else:
        # Generic non-empty check for other fields (Keywords, etc.)
        valid_count = df[column].apply(lambda x: str(x).lower() not in ['none', 'n/a', 'nan', '', 'not found']).sum()
    return (valid_count / len(df)) * 100


def find_csv_file(model_name, mode):
    """Robust file finder for different naming conventions."""
    platform_safe = PLATFORM_NAME.replace(' ', '_')

    # Priority list of filename patterns
    patterns = [
        f"clash_{model_name}_{mode}_{platform_safe}.csv",
        f"clash_{model_name}_{platform_safe}.csv" if mode == "cold" else None,
        f"clash_{model_name}_{platform_safe}_{mode}.csv"
    ]

    for pattern in patterns:
        if pattern:
            filepath = os.path.join(LOG_DIR, pattern)
            if os.path.exists(filepath):
                return filepath
    return None


def inspect():
    if not os.path.exists(REG_FILE):
        print(f"❌ Error: Regex baseline missing at {REG_FILE}")
        return

    df_reg = pd.read_csv(REG_FILE)

    print("\n" + "=" * 100)
    print(f"PNEUMA CLASH INSPECTOR (VERIFIED) | Platform: {PLATFORM_NAME}")
    print("=" * 100)

    # Separate summaries for warm and cold
    warm_data = []
    cold_data = []

    for model_config in MODELS_TO_COMPARE:
        model_name = model_config["name"]

        for mode in model_config["modes"]:
            csv_file = find_csv_file(model_name, mode)

            if not csv_file:
                print(f"⚠️ Skipping {model_name} ({mode}): Result file not found.")
                continue

            df_slm = pd.read_csv(csv_file)

            # Filter out ERROR rows (failed processing) to avoid skewing TPS
            df_slm = df_slm[df_slm['title'] != 'ERROR']

            # Merge with regex baseline on filename
            m = pd.merge(df_slm, df_reg, on="filename", suffixes=("_slm", "_reg"))

            # --- METRICS CALCULATION ---
            # 1. Title Similarity (Context Stability Check)
            title_sim = m.apply(lambda r: ratio(str(r.title_slm), str(r.title_reg)), axis=1).mean()

            # 2. VERIFIED DOI Discovery (The "Intelligence" Metric)
            doi_rate_slm = calculate_verified_rate(df_slm, 'doi')

            # 3. Keyword Yield
            kw_rate_slm = calculate_verified_rate(df_slm, 'keywords')

            # 4. Author Delta (Hallucination vs Recall check)
            noise_reduction = df_reg['authors'].str.len().mean() - df_slm['authors'].str.len().mean()

            metric_row = {
                "Model": model_name,
                "Mode": mode.upper(),
                "Avg_TPS": round(df_slm['tps'].mean(), 2),
                "Title_Sim": f"{title_sim:.2f}",
                "Verified_DOI": f"{doi_rate_slm:.1f}%",
                "KW_Yield": f"{kw_rate_slm:.1f}%",
                "Auth_Δ": round(noise_reduction, 1)
            }

            if "WARM" in mode.upper():
                warm_data.append(metric_row)
            else:
                cold_data.append(metric_row)

    # --- WARM MODE REPORT ---
    if warm_data:
        print("\n🔥 WARM MODE PERFORMANCE")
        print("-" * 100)
        warm_df = pd.DataFrame(warm_data)
        print(warm_df.to_string(index=False))

    # --- COLD START REPORT ---
    if cold_data:
        print("\n❄️  COLD START PERFORMANCE")
        print("-" * 100)
        cold_df = pd.DataFrame(cold_data)
        print(cold_df.to_string(index=False))

    # --- BASELINE REFERENCE ---
    print("\n" + "=" * 100)
    print(f"📊 REGEX BASELINE METRICS (Control Group)")
    print("-" * 100)
    # Apply same verification to Regex to be fair
    reg_doi_rate = calculate_verified_rate(df_reg, 'doi')
    print(f"Verified DOI Discovery: {reg_doi_rate:.1f}%")
    print(f"Keyword Yield:          {calculate_verified_rate(df_reg, 'keywords'):.1f}%")
    print("-" * 100)
    print("Note: 'Verified_DOI' excludes placeholders like '10.1145/nnnnnnn' and hallucinations.")
    print("=" * 100 + "\n")

    # --- COMPARATIVE DELTA ANALYSIS ---
    if warm_data and cold_data:
        print("🔬 ARCHITECTURAL PARADOX CHECK (Warm vs. Cold Delta)")
        print("-" * 100)

        all_data = warm_data + cold_data
        df_all = pd.DataFrame(all_data)

        for model_name in df_all['Model'].unique():
            subset = df_all[df_all['Model'] == model_name]
            if len(subset) >= 2:
                # Extract float values for calculation
                warm_row = subset[subset['Mode'].str.contains('WARM')]
                cold_row = subset[~subset['Mode'].str.contains('WARM')]

                if not warm_row.empty and not cold_row.empty:
                    w_tps = float(warm_row.iloc[0]['Avg_TPS'])
                    c_tps = float(cold_row.iloc[0]['Avg_TPS'])

                    # Calculate % change
                    delta = ((c_tps - w_tps) / w_tps) * 100

                    # Paradox Condition: Cold is FASTER than Warm
                    is_paradox = delta > 0
                    icon = "⚡ PARADOX" if is_paradox else "📉 Standard"

                    print(
                        f"{model_name:12s} | Warm: {w_tps:5.2f} TPS | Cold: {c_tps:5.2f} TPS | Δ: {delta:+.1f}%  [{icon}]")
        print("=" * 100 + "\n")


if __name__ == "__main__":
    inspect()