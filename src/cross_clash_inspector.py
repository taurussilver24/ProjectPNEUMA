import pandas as pd
import os
import re
from Levenshtein import ratio

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# The platforms we want to hunt for (matches your filenames)
PLATFORMS = [
    "Dimensity_9300+(Vulkan)",
    "Apple_M1(Metal)",
    "AMD_RX_6800_(Vulkan)"
]

# Models to look for
MODELS_TO_COMPARE = [
    {"name": "Qwen2.5-3B", "modes": ["warm", "cold_start"]},
    {"name": "Phi-4", "modes": ["warm", "cold_start"]},
    {"name": "LFM-2.5", "modes": ["warm", "cold"]}
]

# --- 2. INTEGRITY FILTERS (Your "Hallucination Shield") ---
DOI_BLACKLIST = [
    "nnnnn", "xxxx", "example", "1234", "not found", "not provided",
    "to be determined", "futuredoi", "doi_string", "yoursite", "abcd123",
    "insert doi", "doi:", "pmlr"
]


def is_verified_doi(doi_text):
    """Rigorous check for real DOIs vs. SLM hallucinations."""
    if not doi_text or pd.isna(doi_text): return False
    val = str(doi_text).lower().strip()

    if any(p in val for p in DOI_BLACKLIST): return False
    if re.search(r'[nx]{3,}', val): return False  # Detect nnnnnnn

    # Standard DOI Regex
    doi_pattern = r'^10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+$'
    if re.match(doi_pattern, val): return True

    if "doi.org/10." in val:
        if "xxxx" in val or "nnnn" in val: return False
        return True

    return False


def calculate_verified_rate(df, column):
    if column == 'doi':
        valid_count = df[column].apply(is_verified_doi).sum()
    else:
        valid_count = df[column].apply(lambda x: str(x).lower() not in ['none', 'n/a', 'nan', '', 'not found']).sum()
    return (valid_count / len(df)) * 100


def find_csv_file(platform, model_name, mode):
    """Finds the log file for a specific Platform + Model + Mode combo."""
    # Handle filename inconsistencies (spaces vs underscores)
    plat_clean = platform.replace(" ", "_")

    patterns = [
        f"clash_{model_name}_{mode}_{plat_clean}.csv",
        f"clash_{model_name}_{plat_clean}_{mode}.csv",
        # Some LFM files might just be 'clash_LFM_Dimensity.csv' for cold
        f"clash_{model_name}_{plat_clean}.csv" if "cold" in mode else None
    ]

    for p in patterns:
        if p and os.path.exists(os.path.join(LOG_DIR, p)):
            return os.path.join(LOG_DIR, p)
    return None


def find_baseline(platform):
    """Finds the Regex Baseline for the platform."""
    plat_clean = platform.replace(" ", "_")
    path = os.path.join(LOG_DIR, f"clash_REGEX_BASELINE_{plat_clean}.csv")
    if os.path.exists(path):
        return path
    # Fallback: Use ANY regex baseline if specific one missing (Control group is static)
    fallback = os.path.join(LOG_DIR, "clash_REGEX_BASELINE_Apple_M1(Metal).csv")
    if os.path.exists(fallback):
        return fallback
    return None


# --- 3. MAIN INSPECTION LOOP ---
def run_cross_platform_inspection():
    print("====================================================================================================")
    print("⚔️  PNEUMA CROSS-PLATFORM INSPECTOR (Veritas Edition)")
    print("====================================================================================================")

    leaderboard = []

    for platform in PLATFORMS:
        print(f"\n🖥️  ANALYZING PLATFORM: {platform}")
        print("-" * 100)

        # 1. Load Baseline (Control Group)
        reg_file = find_baseline(platform)
        if not reg_file:
            print(f"   ⚠️ No Regex Baseline found. Skipping accuracy checks.")
            df_reg = None
        else:
            df_reg = pd.read_csv(reg_file)

        for model_config in MODELS_TO_COMPARE:
            model_name = model_config["name"]

            for mode in model_config["modes"]:
                csv_file = find_csv_file(platform, model_name, mode)

                if not csv_file:
                    continue  # Skip silently to keep output clean

                try:
                    df_slm = pd.read_csv(csv_file)
                    df_slm = df_slm[df_slm['title'] != 'ERROR']  # Filter crashes

                    # Merge for Accuracy Checks
                    title_sim = 0.0
                    if df_reg is not None:
                        # Clean merge on filename
                        m = pd.merge(df_slm, df_reg, on="filename", suffixes=("_slm", "_reg"))
                        if not m.empty:
                            title_sim = m.apply(lambda r: ratio(str(r.title_slm), str(r.title_reg)), axis=1).mean()

                    # Calculate Metrics
                    tps = df_slm['tps'].mean()
                    doi_acc = calculate_verified_rate(df_slm, 'doi')

                    # Add to Leaderboard
                    leaderboard.append({
                        "Platform": platform,
                        "Model": model_name,
                        "Mode": "WARM" if "warm" in mode.lower() else "COLD",
                        "TPS": tps,
                        "DOI_Acc": doi_acc,
                        "Title_Sim": title_sim * 100
                    })

                    print(f"   ✅ Processed {model_name} ({mode}): {tps:.1f} t/s | DOI Verified: {doi_acc:.1f}%")

                except Exception as e:
                    print(f"   ❌ Error reading {csv_file}: {e}")

    # --- 4. FINAL LEADERBOARD ---
    if not leaderboard:
        print("\n❌ No valid logs found for any platform.")
        return

    df = pd.DataFrame(leaderboard)

    print("\n" + "=" * 100)
    print("🏆 FINAL SCOREBOARD (Speed vs. Intelligence)")
    print("=" * 100)

    # Split by Mode for fair comparison
    for mode in ["WARM", "COLD"]:
        subset = df[df["Mode"] == mode].sort_values(by="TPS", ascending=False)

        if subset.empty: continue

        print(f"\n📊 {mode} START RANKING")
        print(f"{'Platform':<25} {'Model':<12} {'TPS (Speed)':<12} {'DOI % (Intel)':<12} {'Context %':<10}")
        print("-" * 80)

        for _, row in subset.iterrows():
            print(
                f"{row['Platform']:<25} {row['Model']:<12} {row['TPS']:<12.1f} {row['DOI_Acc']:<12.1f} {row['Title_Sim']:<10.1f}")

    print("-" * 80)

    # Identify the "Balanced Champion" (Best Speed * Accuracy)
    # We define Score = TPS * (DOI_Acc / 100)
    df["Composite_Score"] = df["TPS"] * (df["DOI_Acc"] / 100)
    winner = df.loc[df["Composite_Score"].idxmax()]

    print(f"\n👑 OVERALL CHAMPION: {winner['Platform']} running {winner['Model']}")
    print(f"   (Best balance of Speed {winner['TPS']:.1f} t/s & Accuracy {winner['DOI_Acc']:.1f}%)")


if __name__ == "__main__":
    run_cross_platform_inspection()