import os, glob
import pandas as pd
import numpy as np

RESULTS_FOLDER = "results"
OUT_SUMMARY = "summary_per_instance_qlearning_egreedy.csv"

def summarize_results(results_folder: str = RESULTS_FOLDER) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_folder, "results_*.csv")))
    records = []

    for fpath in files:
        inst_name = os.path.basename(fpath)[len("results_"):-4]
        df = pd.read_csv(fpath)

        # compute total penalty 
        for col in ["PenT", "PenH", "PenS"]:
            if col not in df.columns:
                df[col] = 0.0
        df["pen_total"] = df["PenT"] + df["PenH"] + df["PenS"]

        #  find best run 
        feasible_mask = df["pen_total"] == 0
        if feasible_mask.any():
            # at least one feasible: pick best f among them
            best_idx = df.loc[feasible_mask, "f"].idxmin()
            best_pen = 0.0
            avg_f    = df.loc[feasible_mask, "f"].mean()
        else:
            # no feasible: pick lowest penalty, break ties on f
            best_idx = df.sort_values(["pen_total", "f"], ascending=[True, True]).index[0]
            best_pen = df.loc[best_idx, "pen_total"]
            min_pen  = df["pen_total"].min()
            avg_f    = df.loc[df["pen_total"] == min_pen, "f"].mean()

        records.append({
            "instance":       inst_name,
            "best_f":         df.loc[best_idx, "f"],
            "best_penalty":   best_pen,
            "PenT":           df.loc[best_idx, "PenT"],
            "PenH":           df.loc[best_idx, "PenH"],
            "PenS":           df.loc[best_idx, "PenS"],
            "time_s":         df.loc[best_idx, "time_s"],
            "best_time_s":    df.loc[best_idx, "best_time_s"]
                               if "best_time_s" in df.columns else np.nan,
            "avg_best_time_s": df["best_time_s"].mean()
                                if "best_time_s" in df.columns else np.nan,
            "avg_f":          avg_f
        })

    df_summary = pd.DataFrame(records).sort_values("instance").reset_index(drop=True)
    out_path = os.path.join(results_folder, OUT_SUMMARY)
    df_summary.to_csv(out_path, index=False)
    print(f"Saved per-instance summary -> {out_path}")
    return df_summary

if __name__ == "__main__":
    summarize_results()