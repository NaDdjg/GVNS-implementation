import os
import sys
import pandas as pd
import glob

def merge_summaries(input_dir: str, output_file: str = "merged_summary.csv"):
    """
    Scans input_dir for files ending with *_summary.csv,
    merges them with paper results, and keeps only the best run (min best_f) per instance.
    """

    summary_files = glob.glob(os.path.join(input_dir, "**", "*_summary.csv"), recursive=True)

    if not summary_files:
        print("[ERROR] No *_summary.csv files found in:", input_dir)
        return

    all_rows = []

    for path in summary_files:
        try:
            df = pd.read_csv(path)

            # Normalize column names (best_F → best_f)
            if "best_F" in df.columns and "best_f" not in df.columns:
                df = df.rename(columns={"best_F": "best_f"})

            if "instance" not in df.columns:
                print(f"[WARNING] File {path} has no 'instance' column. Skipped.")
                continue

            if "best_f" not in df.columns:
                print(f"[WARNING] File {path} has no 'best_f'. Using entire file anyway.")
                # still accept the file; best_f filtering will fallback later

            # If input_dir refers to Bredstrom, convert best_f from minutes to hours
            if "bredstrom" in input_dir.replace("\\", "/").lower():
                if "best_f" in df.columns:
                    df["best_f"] = pd.to_numeric(df["best_f"], errors="coerce") / 60

            all_rows.append(df)

        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")

    if not all_rows:
        print("[ERROR] No valid summary files parsed.")
        return

    merged = pd.concat(all_rows, ignore_index=True)

    # Convert best_f and time to numeric if possible
    if "best_f" in merged.columns:
        merged["best_f"] = pd.to_numeric(merged["best_f"], errors="coerce")
        merged = merged.rename(columns={"best_f": "f_method"})

        # Convert time to numeric if it exists
        if "time" in merged.columns:
            merged["time"] = pd.to_numeric(merged["time"], errors="coerce")
        
        # Calculate averages per instance across all runs
        averages = merged.groupby("instance")[["f_method", "time"]].mean()
        averages = averages.rename(columns={"f_method": "avg_f", "time": "avg_time"})
        
        # Keep best run per instance (minimum best_f)
        merged = merged.sort_values(by=["instance", "f_method"])
        merged = merged.groupby("instance", as_index=False).first()
        
        
        # Merge averages back with best run data
        merged = merged.merge(averages, left_on="instance", right_index=True, how="left")
    else:
        # No best_f column found anywhere → keep first row for each instance
        merged = merged.groupby("instance", as_index=False).first()

    # Read paper results
    papers_file = "results/reference/GVNS_reimplementation.csv"
    try:
        papers_df = pd.read_csv(papers_file)
    except Exception as e:
        print(f"[ERROR] Failed to read papers results from {papers_file}: {e}")
        return

    # Merge with paper results
    merged = merged.merge(papers_df, on="instance", how="left")

    # Select and reorder columns (in specific order)
    columns_to_keep = ["instance", "run", "f_method", "time", "f_GVNS", "time_GVNS", "avg_f", "avg_time"]
    merged = merged[[col for col in columns_to_keep if col in merged.columns]]

    # Save output
    merged.to_csv(output_file, index=False)
    print(f"[OK] Merged summary saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_summaries.py <results_folder> [output_file]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = "result_summaries/merged_summary_First.csv" if len(sys.argv) < 3 else sys.argv[2]

    merge_summaries(input_dir, output_file)