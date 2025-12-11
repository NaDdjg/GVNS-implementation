import os
import pandas as pd
from glob import glob

def collect_neigh_usage(results_dir="results/neigh_usage"):
    """
    Reads all *_neigh_usage.csv files and aggregates global neighbor usage stats.
    Outputs: neigh_usage_global.csv
    """
    pattern = os.path.join(results_dir, "*_neigh_usage.csv")
    files = glob(pattern)

    if not files:
        print("No neigh_usage files found.")
        return

    all_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            instance = os.path.basename(f).replace("_neigh_usage.csv", "")
            df["instance"] = instance
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Combine everything
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Aggregate usage by neighborhood
    global_stats = (
        full_df.groupby("neighborhood")["count"]
        .sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )

    # Save global stats
    out_file = os.path.join(results_dir, "neigh_usage_global.csv")
    global_stats.to_csv(out_file, index=False)

    print(f"Saved global neighbor usage statistics to: {out_file}")

    # Optionally: also provide per-instance %, normalized
    percent_stats = (
        full_df.groupby(["instance", "neighborhood"])["count"]
        .sum()
        .groupby(level=0)
        .apply(lambda s: s / s.sum())
        .reset_index(name="percentage")
    )

    out_file_pct = os.path.join(results_dir, "neigh_usage_percentages.csv")
    percent_stats.to_csv(out_file_pct, index=False)
    print(f"Saved per-instance percentage usage to: {out_file_pct}")


if __name__ == "__main__":
    collect_neigh_usage() 
