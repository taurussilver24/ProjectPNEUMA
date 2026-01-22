import pandas as pd
import os
from Levenshtein import ratio

# --- 1. INSPECTION CONFIGURATION ---
PLATFORM_NAME = "AMD RX 6800 (Vulkan)"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Model configurations with their file patterns
MODELS_TO_COMPARE = [
    {"name": "Qwen2.5-3B", "modes": ["warm", "cold_start"]},
    {"name": "Phi-4", "modes": ["warm", "cold_start"]},
    {"name": "LFM-2.5", "modes": ["warm", "cold"]}  # Note: LFM uses "cold" not "cold_start"
]

REG_FILE = os.path.join(LOG_DIR, f"clash_REGEX_BASELINE_{PLATFORM_NAME.replace(' ', '_')}.csv")


def calculate_discovery_rate(df, column):
    """Calculates % of rows where data is NOT 'None' or 'N/A'"""
    valid_count = df[column].apply(lambda x: str(x).lower() not in ['none', 'n/a', 'nan', '']).sum()
    return (valid_count / len(df)) * 100


def find_csv_file(model_name, mode):
    """Find the CSV file for a given model and mode"""
    platform_safe = PLATFORM_NAME.replace(' ', '_')

    # Try different naming patterns
    patterns = [
        f"clash_{model_name}_{mode}_{platform_safe}.csv",
        f"clash_{model_name}_{platform_safe}.csv" if mode == "cold" else None
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

    print("\n" + "=" * 90)
    print(f"PNEUMA CLASH INSPECTOR | Platform: {PLATFORM_NAME}")
    print("=" * 90)

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

            # Add mode column if missing (for old warm CSVs)
            if 'mode' not in df_slm.columns:
                df_slm['mode'] = mode

            # Merge with regex baseline
            m = pd.merge(df_slm, df_reg, on="filename", suffixes=("_slm", "_reg"))

            # --- METRICS ---
            # 1. Title Similarity
            title_sim = m.apply(lambda r: ratio(str(r.title_slm), str(r.title_reg)), axis=1).mean()

            # 2. DOI Discovery
            doi_rate_slm = calculate_discovery_rate(df_slm, 'doi')

            # 3. Keyword Yield
            kw_rate_slm = calculate_discovery_rate(df_slm, 'keywords')

            # 4. Author Delta
            noise_reduction = df_reg['authors'].str.len().mean() - df_slm['authors'].str.len().mean()

            metric_row = {
                "Model": model_name,
                "Mode": mode.upper(),
                "Avg_TPS": round(df_slm['tps'].mean(), 2),
                "Title_Sim": f"{title_sim:.2f}",
                "DOI_Disc": f"{doi_rate_slm:.1f}%",
                "KW_Yield": f"{kw_rate_slm:.1f}%",
                "Auth_Δ": round(noise_reduction, 1)
            }

            # Categorize by mode
            if "warm" in mode.lower():
                warm_data.append(metric_row)
            else:
                cold_data.append(metric_row)

    # --- WARM MODE REPORT ---
    if warm_data:
        print("\n🔥 WARM MODE PERFORMANCE")
        print("-" * 90)
        warm_df = pd.DataFrame(warm_data)
        print(warm_df.to_string(index=False))

    # --- COLD START REPORT ---
    if cold_data:
        print("\n❄️  COLD START PERFORMANCE")
        print("-" * 90)
        cold_df = pd.DataFrame(cold_data)
        print(cold_df.to_string(index=False))

    # --- BASELINE REFERENCE ---
    print("\n" + "=" * 90)
    print(f"📊 REGEX BASELINE METRICS")
    print("-" * 90)
    print(f"DOI Discovery:     {calculate_discovery_rate(df_reg, 'doi'):.1f}%")
    print(f"Keyword Yield:     {calculate_discovery_rate(df_reg, 'keywords'):.1f}%")
    print(f"Avg Author Length: {df_reg['authors'].str.len().mean():.1f} chars")
    print("-" * 90)
    print("Note: Negative Auth_Δ suggests SLMs found MORE authors (higher recall)")
    print("=" * 90 + "\n")

    # --- COMPARATIVE ANALYSIS ---
    if warm_data and cold_data:
        print("🔬 WARM vs COLD COMPARISON")
        print("-" * 90)

        # Group by model and compare
        all_data = warm_data + cold_data
        df_all = pd.DataFrame(all_data)

        for model_name in df_all['Model'].unique():
            model_subset = df_all[df_all['Model'] == model_name]
            if len(model_subset) >= 2:
                warm_row = model_subset[model_subset['Mode'].str.contains('WARM')]
                cold_row = model_subset[~model_subset['Mode'].str.contains('WARM')]

                if not warm_row.empty and not cold_row.empty:
                    warm_tps = warm_row.iloc[0]['Avg_TPS']
                    cold_tps = cold_row.iloc[0]['Avg_TPS']
                    overhead = ((warm_tps - cold_tps) / cold_tps) * 100

                    status = "✅" if overhead > 0 else "⚠️"
                    print(
                        f"{status} {model_name:15s} | Warm: {warm_tps:6.2f} TPS | Cold: {cold_tps:6.2f} TPS | Δ: {overhead:+.1f}%")

        print("=" * 90 + "\n")


if __name__ == "__main__":
    inspect()