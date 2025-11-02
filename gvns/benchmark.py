import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from gvns.instance import Instance
from gvns.gvns import run_gvns
import random
import numpy as np

# --------------------------------------------------------------
# Run GVNS on one instance
# --------------------------------------------------------------
def run_instance(path, n_runs=1, params=None, results_dir="results"):
    """Run GVNS multiple times on one instance, save traces and plots."""
    name = os.path.basename(path).split(".")[0]
    print(f"\n=== Running instance {name} ===")

    random.seed(params.get("seed", 42))
    np.random.seed(params.get("seed", 42))

    inst = Instance().load_liu_format(path)
    best_overall = None
    results = []

    for run_id in range(1, n_runs + 1):
        print(f"\n[RUN {run_id}/{n_runs}] Starting GVNS on {name} ...")

        start = time.time()
        trace = []

        # ----------------------------------------------------------
        # Run GVNS with trace logging
        # ----------------------------------------------------------
        best = run_gvns(inst, params=params, verbose=True, trace_list=trace)
        elapsed = time.time() - start

        # Save trace as CSV
        df = pd.DataFrame(trace, columns=["iter", "F", "f", "PenT", "PenH", "PenS", "elapsed"])
        csv_path = os.path.join(results_dir, f"{name}_run{run_id}.csv")
        df.to_csv(csv_path, index=False)

        # Record final stats
        results.append({
            "instance": name,
            "run": run_id,
            "best_F": best.Fval,
            "best_f": best.travel_time,
            "PenT": best.penT,
            "PenH": best.penH,
            "PenS": best.penS,
            "time": elapsed
        })

        if best_overall is None or best.travel_time < best_overall.travel_time:
            best_overall = best.copy()

        # ----------------------------------------------------------
        # Plot evolution of f (travel cost)
        # ----------------------------------------------------------
        plt.figure(figsize=(8, 5))
        plt.title(f"GVNS evolution for {name} (run {run_id})")
        plt.xlabel("Iteration")
        plt.ylabel("Objective f (travel distance)")
        plt.scatter(df["iter"], df["f"], s=20, label="raw f", color="blue", alpha=0.6)
        plt.plot(df["iter"], df["f"].cummin(), label="best f", color="orange", linewidth=2)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(results_dir, f"{name}_run{run_id}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[RUN {run_id}] Done. Best f={best.travel_time:.3f}, PenS={best.penS:.1f}, time={elapsed:.2f}s")

    # ----------------------------------------------------------
    # Summary over all runs
    # ----------------------------------------------------------
    df_res = pd.DataFrame(results)
    summary_path = os.path.join(results_dir, f"{name}_summary.csv")
    df_res.to_csv(summary_path, index=False)

    print(f"\n=== Summary for {name} ===")
    print(df_res[["run", "best_f", "PenS", "time"]])
    print(f"→ mean f={df_res['best_f'].mean():.3f}, best f={df_res['best_f'].min():.3f}")

    return df_res


# --------------------------------------------------------------
# Run GVNS on all .txt instances in a folder
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# Callable main for run_benchmark.py
# --------------------------------------------------------------
def benchmark_main(INST_FOLDER, RESULTS_FOLDER, PARAMS, N_RUNS=1):
    """Main callable for the root run_benchmark.py."""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    run_folder(folder=INST_FOLDER, n_runs=N_RUNS, params=PARAMS, results_dir=RESULTS_FOLDER)


# --------------------------------------------------------------
# Local run (for standalone testing)
# --------------------------------------------------------------
# if __name__ == "__main__":
#     INST_FOLDER = "instances"
#     RESULTS_FOLDER = "results"
#     PARAMS = {
#         "contmax": 100,
#         "p_init": 1,
#         "p_max": 10,
#         "Dp": 0.3,
#         "w1_init": 0.5,
#         "w2_init": 0.1,
#         "alpha": 0.1,
#         "M": 1e6,
#         "seed": 42,
#         "Tmax": 600
#     }
#     N_RUNS = 1
#     benchmark_main(INST_FOLDER, RESULTS_FOLDER, PARAMS, N_RUNS)
