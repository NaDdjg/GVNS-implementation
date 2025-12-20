import os
import pandas as pd
from glob import glob

def collect_neigh_usage(results_dir="result_summaries/neigh_usage", output_dir="result_summaries/neigh_usage"):
    """
    Reads all *_neigh_usage.csv files and aggregates global neighbor usage stats.
    Outputs: neigh_usage_global.csv
    """
    pattern = os.path.join(results_dir, "*_usage.csv")
    files = glob(pattern)

    os.makedirs(output_dir, exist_ok=True)

    if not files:
        print("No neigh_usage files found.")
        return

    all_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            instance = os.path.basename(f).replace("_usage.csv", "")
            df["instance"] = instance
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Combine everything
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Aggregate usage by neighborhood
    global_stats = (
        full_df.groupby("Neighborhood_ID")["Usage_Count"]
        .sum()
        .reset_index()
        .sort_values("Usage_Count", ascending=False)
    )

    # Calculate total usage for percentages
    total_usage = global_stats["Usage_Count"].sum()
    global_stats["percentage"] = global_stats["Usage_Count"] / total_usage

    # Save global stats with percentage column
    out_file = os.path.join(output_dir, "neigh_usage_global.csv")
    global_stats.to_csv(out_file, index=False)

    print(f"Saved global neighbor usage statistics to: {out_file}")

    # Remove the separate percentage file saving


if __name__ == "__main__":
    collect_neigh_usage(
        results_dir="results/without RL/reverse_order/Best/neigh_usage", 
        output_dir="result_summaries/neigh_usage/Best_reverse"
    ) 
