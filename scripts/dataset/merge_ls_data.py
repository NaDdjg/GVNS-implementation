import os
import glob
import pandas as pd


# ============================================================
# Configuration
# ============================================================
BASE_DIR = "gvns/ls_data"
FI_DIR = os.path.join(BASE_DIR, "FI")
BI_DIR = os.path.join(BASE_DIR, "BI")

OUTPUT_PATH = os.path.join(BASE_DIR, "merged_ls_decisions.csv")


# ============================================================
# Helper: load all CSVs from a directory
# ============================================================
def load_mode_data(mode_dir: str, strategy_label: int):
    """
    Load all CSV files from a given directory (FI or BI),
    attach a 'strategy' column, and return a list of DataFrames.

    Parameters
    ----------
    mode_dir : str
        Path to FI or BI directory
    strategy_label : int
        0 for FI, 1 for BI

    Returns
    -------
    list[pd.DataFrame]
    """

    dataframes = []

    csv_files = sorted(glob.glob(os.path.join(mode_dir, "*.csv")))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in {mode_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        # sanity check
        if df.empty:
            print(f"[WARNING] Empty file skipped: {csv_path}")
            continue

        # add strategy label
        df["strategy"] = strategy_label

        # keep track of origin file (useful for debugging)
        df["source_file"] = os.path.basename(csv_path)

        dataframes.append(df)

    return dataframes


# ============================================================
# Main merge logic
# ============================================================
def main():
    print("Loading FI data...")
    fi_dfs = load_mode_data(FI_DIR, strategy_label=0)

    print("Loading BI data...")
    bi_dfs = load_mode_data(BI_DIR, strategy_label=1)

    # --------------------------------------------------------
    # Schema consistency check
    # --------------------------------------------------------
    fi_cols = set(fi_dfs[0].columns)
    bi_cols = set(bi_dfs[0].columns)

    if fi_cols != bi_cols:
        missing_in_bi = fi_cols - bi_cols
        missing_in_fi = bi_cols - fi_cols

        raise RuntimeError(
            "FI / BI column mismatch:\n"
            f"Missing in BI: {missing_in_bi}\n"
            f"Missing in FI: {missing_in_fi}"
        )

    print("Schema check passed.")

    # --------------------------------------------------------
    # Concatenate all rows
    # --------------------------------------------------------
    all_dfs = fi_dfs + bi_dfs
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # --------------------------------------------------------
    # Final sanity checks
    # --------------------------------------------------------
    print("Merged dataset summary:")
    print(merged_df[["strategy"]].value_counts())
    print("Total rows:", len(merged_df))
    print("Total columns:", len(merged_df.columns))

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
