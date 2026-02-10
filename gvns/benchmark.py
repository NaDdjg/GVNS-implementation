import os
import time
import random
import numpy as np # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore

import multiprocessing as mp
from functools import partial

from gvns.instance import Instance
from gvns.gvns import run_gvns

# profiling 
import cProfile
import pstats
from line_profiler import LineProfiler

# run GVNS on one instance
def run_instance(
        path, 
        n_runs=1, 
        params=None, 
        agent=None,
        first_improvement = True,
        results_dir="results",
        qtables_dir=None
    ):
    """run GVNS multiple times on one instance, save traces and plots."""
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

        # new agent per run (no leakage)
        run_agent = None
        if agent is not None:
            run_agent = agent.clone() if hasattr(agent, "clone") else agent

        start = time.time()
        trace = []

        # Run GVNS (classic or RL)
        best = run_gvns(
            inst, 
            params=params, 
            agent=run_agent, 
            first_improvement = first_improvement,
            verbose=True, 
            trace_list=trace, 
            instance_name = name,
            qtables_dir=qtables_dir
        )
        elapsed = time.time() - start

        # Build trace DataFrame
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

# ============================================================
# Profiling functions
# ============================================================
def profile_run_instance(*args, **kwargs):
    lp = LineProfiler()

    # Entry points
    lp.add_function(run_instance)

    from gvns.gvns import run_gvns
    lp.add_function(run_gvns)

    import gvns.local_search as ls
    import gvns.operators as ops
    import gvns.solution as solution
    import gvns.search_state.search_state as ss

    lp.add_module(ls)
    lp.add_module(ops)
    lp.add_module(solution)
    lp.add_module(ss)

    lp.runctx(
        "run_instance(*args, **kwargs)",
        globals(),
        locals()
    )

    out_dir = os.path.join("result_summaries", "debug")
    os.makedirs(out_dir, exist_ok=True)

    # optional: nice filename
    path = kwargs.get("path")
    instance_name = os.path.basename(path).split(".")[0]

    mode = "FI" if kwargs.get("first_improvement", True) else "BI"
    out_path = os.path.join(out_dir, f"line_profile_{instance_name}_{mode}.txt")

    with open(out_path, "w") as f:
        lp.print_stats(stream=f, output_unit=1e-3)  # ms

def profile_run_instance_cprofile(*args, **kwargs):
    """
    Function-level profiler using cProfile + pstats.
    Profiles the full GVNS execution path.
    """

    # Prepare output directory
    out_dir = os.path.join("result_summaries", "debug")
    os.makedirs(out_dir, exist_ok=True)

    path = kwargs.get("path")
    instance_name = os.path.basename(path).split(".")[0]

    mode = "FI" if kwargs.get("first_improvement", True) else "BI"

    prof_path = os.path.join(out_dir, f"cprofile_{instance_name}_{mode}.prof")
    txt_path  = os.path.join(out_dir, f"cprofile_{instance_name}_{mode}.txt")

    # Run profiler
    profiler = cProfile.Profile()
    profiler.enable()

    result = run_instance(*args, **kwargs)

    profiler.disable()

    # Save binary profile 
    profiler.dump_stats(prof_path)

    # Save summary
    with open(txt_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)

        stats.sort_stats("cumulative")   
        stats.print_stats()           

        # f.write("\n\nCALLERS (who calls expensive functions)\n")
        # stats.print_callers(20)

        # f.write("\n\nCALLEES (what expensive functions call)\n")
        # stats.print_callees(20)

    return result

# ============================================================
# Run folder
# ============================================================
# helper for multiprocessing
def _run_one_instance_mp(
    path_full,
    n_runs,
    params,
    agent,
    first_improvement,
    results_dir,
    qtables_dir
):
    return run_instance(
        path_full,
        n_runs=n_runs,
        params=params,
        agent=agent,
        first_improvement=first_improvement,
        results_dir=results_dir,
        qtables_dir=qtables_dir
    )

# classic sequential runner
def run_folder(folder, n_runs=1, params=None, agent=None, first_improvement = True, results_dir="results", qtables_dir=None):
    """Run all instances in the specified folder sequentially."""
    paths = sorted([p for p in os.listdir(folder) if p.endswith(".txt")])
    if not paths:
        print(f"No .txt instances found in {folder}")
        return

    all_results = []
    for p in paths:
        path_full = os.path.join(folder, p)
        res = run_instance(
            path_full, 
            n_runs=n_runs, 
            params=params, 
            agent=agent,
            first_improvement = first_improvement,
            results_dir=results_dir,
            qtables_dir=qtables_dir
        )
        all_results.append(res)

    # combine all summaries into one global summary CSV
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(os.path.join(results_dir, "all_instances_summary.csv"), index=False)
    print("\nBenchmark completed.")
    print(df_all.groupby("instance")["best_f"].agg(["mean", "std", "min", "max"]))

# profiling version for one instance only
def run_folder_profiling(folder, n_runs=1, params=None, agent=None, first_improvement = True, results_dir="results", qtables_dir=None):
    """Run all instances in the specified folder sequentially."""
    paths = sorted([p for p in os.listdir(folder) if p.endswith(".txt")])
    if not paths:
        print(f"No .txt instances found in {folder}")
        return

    all_results = []
    # line_profiler version (line-by-line)
    # res = profile_run_instance(
    #     path="instances\\30\\C102_30_3.txt", 
    #     n_runs=n_runs, 
    #     params=params,             
    #     agent=agent,
    #     first_improvement = first_improvement,
    #     results_dir=results_dir,
    #     qtables_dir=qtables_dir
    # )

    # cprofile version (function-level)
    res = profile_run_instance_cprofile(
        path="instances\\30\\C102_30_3.txt", 
        n_runs=n_runs, 
        params=params,             
        agent=agent,
        first_improvement = first_improvement,
        results_dir=results_dir,
        qtables_dir=qtables_dir,
    )
    all_results.append(res)

    # combine all summaries into one global summary CSV
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(os.path.join(results_dir, "all_instances_summary.csv"), index=False)
    print("\nBenchmark completed.")
    print(df_all.groupby("instance")["best_f"].agg(["mean", "std", "min", "max"]))

# multiprocessing version
def run_folder_parallel(
    folder,
    n_runs=1,
    params=None,
    agent=None,
    first_improvement=True,
    results_dir="results",
    qtables_dir=None,
    n_processes=None  
):
    """Run all instances in parallel (one process per instance)."""

    paths = sorted([p for p in os.listdir(folder) if p.endswith(".txt")])
    if not paths:
        print(f"No .txt instances found in {folder}")
        return

    os.makedirs(results_dir, exist_ok=True)
    full_paths = [os.path.join(folder, p) for p in paths]

    worker = partial(
        _run_one_instance_mp,
        n_runs=n_runs,
        params=params,
        agent=agent,
        first_improvement=first_improvement,
        results_dir=results_dir,
        qtables_dir=qtables_dir,
    )

    n_proc = n_processes or mp.cpu_count()
    print(f"Running {len(full_paths)} instances in parallel on {n_proc} processes")

    pool = mp.Pool(processes=n_proc)

    try:
        all_results = pool.map(worker, full_paths)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, terminating workers...")
        pool.terminate()   # kill workers immediately
        pool.join()
        print("All worker processes terminated.")
        return
    else:
        pool.close()       # normal shutdown
        pool.join()

    # combine all summaries into one global summary CSV
    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(
        os.path.join(results_dir, "all_instances_summary.csv"),
        index=False
    )

    print("\nBenchmark completed.")
    print(df_all.groupby("instance")["best_f"].agg(["mean", "std", "min", "max"]))

# ============================================================
# Main benchmark function
# ============================================================
def benchmark_main(INST_FOLDER, RESULTS_FOLDER, QTABLE_DIR, PARAMS, N_RUNS = 1, agent = None, first_improvement = True):
    # check again if folder exists 
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # profiler version for one instance only
    # run_folder_profiling(
    #     folder=INST_FOLDER,
    #     n_runs=N_RUNS,
    #     params=PARAMS,
    #     agent=agent,
    #     first_improvement=first_improvement,
    #     results_dir=RESULTS_FOLDER,            
    #     qtables_dir=QTABLE_DIR,
    # )

    # parallel version
    run_folder_parallel(
        folder=INST_FOLDER,
        n_runs=N_RUNS,
        params=PARAMS,
        agent=agent,
        first_improvement=first_improvement,
        results_dir=RESULTS_FOLDER,            
        qtables_dir=QTABLE_DIR,
        n_processes=18
    )

    # sequential version
    # run_folder(
    #     folder=INST_FOLDER,
    #     n_runs=N_RUNS,
    #     params=PARAMS,
    #     agent=agent,
    #     first_improvement=first_improvement,
    #     results_dir=RESULTS_FOLDER,            
    #     qtables_dir=QTABLE_DIR,
    # )
