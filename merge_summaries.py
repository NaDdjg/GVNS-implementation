import os
import sys
import pandas as pd

def merge_summaries(input_dir: str, output_file: str = "merged_summary.csv"):
    """
    Scans input_dir for files ending with *_summary.csv,
    merges them, and keeps only the best run (min best_f) per instance.
    """

    summary_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith("_summary.csv")
    ]

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

            all_rows.append(df)

        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")

    if not all_rows:
        print("[ERROR] No valid summary files parsed.")
        return

    merged = pd.concat(all_rows, ignore_index=True)

    # Convert best_f to numeric if possible
    if "best_f" in merged.columns:
        merged["best_f"] = pd.to_numeric(merged["best_f"], errors="coerce")

        # Keep best run per instance (minimum best_f)
        merged = merged.sort_values(by=["instance", "best_f"])
        merged = merged.groupby("instance", as_index=False).first()
    else:
        # No best_f column found anywhere → keep first row for each instance
        merged = merged.groupby("instance", as_index=False).first()

    # Save output
    merged.to_csv(output_file, index=False)
    print(f"[OK] Merged summary saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_summaries.py <results_folder> [output_file]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = "merged_summary_qlearning-First.csv" if len(sys.argv) < 3 else sys.argv[2]

    merge_summaries(input_dir, output_file)