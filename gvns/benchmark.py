import os
import time
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from gvns.instance import Instance
from gvns.gvns import run_gvns
import random
import numpy as np # type: ignore

# run GVNS on one instance
def run_instance(path, n_runs=1, params=None, results_dir="results"):
    """Run GVNS multiple times on one instance, save traces and plots."""
    name = os.path.basename(path).split(".")[0]
    print(f"\n=== Running instance {name} ===")

    seed = params.get("seed", 42) if params else 42
    random.seed(seed)
    np.random.seed(seed)

    inst = Instance().load_liu_format(path)
    best_overall = None
    results = []

    for run_id in range(1, n_runs + 1):
        print(f"\n[RUN {run_id}/{n_runs}] Starting GVNS on {name} ...")

        start = time.time()
        trace = []

        # run GVNS with trace logging
        best = run_gvns(inst, params=params, verbose=True, trace_list=trace)
        elapsed = time.time() - start

        # normalize trace rows (handle variable-length rows)
        # if trace is empty, produce an empty DataFrame with canonical columns
        if not trace:
            cols = ["iter", "F", "f", "PenT", "PenH", "PenS", "elapsed"]
            df = pd.DataFrame(columns=cols)
        else:
            # determine max row length
            max_len = max(len(r) for r in trace)
            # pad shorter rows with None
            normalized = [list(r) + [None] * (max_len - len(r)) for r in trace]

            if max_len == 7:
                cols = ["iter", "F", "f", "PenT", "PenH", "PenS", "elapsed"]
            elif max_len == 8:
                cols = ["iter", "F", "f", "PenT", "PenH", "PenS", "elapsed", "best_feasible_f"]
            else:
                # fallback generic column names
                cols = [f"col{i}" for i in range(max_len)]
            df = pd.DataFrame(normalized, columns=cols)

        # save trace as CSV
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f"{name}_run{run_id}.csv")
        df.to_csv(csv_path, index=False)

        # attempt to extract best_feasible_f from trace if present
        best_f_feasible = None
        if "best_feasible_f" in df.columns:
            # take the last non-null value
            series = df["best_feasible_f"].dropna()
            if not series.empty:
                best_f_feasible = float(series.iloc[-1])

        # record final stats (best returned object still used)
        results.append({
            "instance": name,
            "run": run_id,
            "best_f": best.travel_time,
            "PenT": best.penT,
            "PenH": best.penH,
            "PenS": best.penS,
            "time": best.best_time
        })

        if best_overall is None or best.travel_time < best_overall.travel_time:
            best_overall = best.copy()

        # plot evolution of f (travel cost) and best feasible f line
        plt.figure(figsize=(8, 5))
        plt.title(f"GVNS evolution for {name} (run {run_id})")
        plt.xlabel("Iteration")
        plt.ylabel("Objective f (travel distance)")

        # raw f points
        if "iter" in df.columns and "f" in df.columns and not df.empty:
            feasible_df = df[(df["PenT"] == 0) & (df["PenH"] == 0) & (df["PenS"] == 0)]
            if not feasible_df.empty:
                plt.scatter(feasible_df["iter"], feasible_df["f"], s=20,
                            label="feasible f (raw accepted)", alpha=0.7)

            # compute feasible-best series: only consider iterations where penalties are zero (if those columns exist)
            best_series = []
            best_val = float("inf")
            for _, row in df.iterrows():
                f = row.get("f", None)
                PenT = row.get("PenT", None)
                PenH = row.get("PenH", None)
                PenS = row.get("PenS", None)
                # update best only when fully feasible (all penalties zero) and f is positive
                if (PenT in (0.0, 0, None) and PenH in (0.0, 0, None) and PenS in (0.0, 0, None)):
                    # if penalties columns are None, we cannot rely on them; treat as unknown -> do not update best
                    if (PenT is None) or (PenH is None) or (PenS is None):
                        best_series.append(best_val if best_val < float("inf") else float("nan"))
                        continue
                    if f is not None and f > 0:
                        if f < best_val:
                            best_val = f
                best_series.append(best_val if best_val < float("inf") else float("nan"))

            plt.plot(df["iter"], best_series, label="best feasible f (so far)", linewidth=2)

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(results_dir, f"{name}_run{run_id}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[RUN {run_id}] Done. Best f={best.travel_time:.3f}, Best feasible f={(best_f_feasible if best_f_feasible is not None else None)}, PenS={best.penS:.1f}, time={elapsed:.2f}s")

    # summary over all runs
    df_res = pd.DataFrame(results)
    summary_path = os.path.join(results_dir, f"{name}_summary.csv")
    df_res.to_csv(summary_path, index=False)

    print(f"\n=== Summary for {name} ===")
    # safe prints even if some keys missing
    cols_to_show = [c for c in ["run", "best_f", "best_feasible_f", "PenS", "time"] if c in df_res.columns]
    print(df_res[cols_to_show])
    print(f"→ mean f={df_res['best_f'].mean():.3f}, best f={df_res['best_f'].min():.3f}")

    return df_res


# run GVNS on all .txt instances in a folder
def run_folder(folder, n_runs=1, params=None, results_dir="results"):
    """Run all instances in the specified folder sequentially."""
    paths = sorted([p for p in os.listdir(folder) if p.endswith(".txt")])
    if not paths:
        print(f"No .txt instances found in {folder}")
        return

    all_results = []
    for p in paths:
        path_full = os.path.join(folder, p)
        res = run_instance(path_full, n_runs=n_runs, params=params, results_dir=results_dir)
        all_results.append(res)

    # Combine all summaries into one global summary CSV
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(os.path.join(results_dir, "all_instances_summary.csv"), index=False)
    print("\nBenchmark completed.")
    print(df_all.groupby("instance")["best_f"].agg(["mean", "std", "min", "max"]))


def benchmark_main(INST_FOLDER, RESULTS_FOLDER, PARAMS, N_RUNS=1):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    run_folder(folder=INST_FOLDER, n_runs=N_RUNS, params=PARAMS, results_dir=RESULTS_FOLDER)
