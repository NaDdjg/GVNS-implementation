import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_it")
os.makedirs(PLOTS_DIR, exist_ok=True)

# pattern to pick up CSV trace files produced by your benchmark
pattern = os.path.join(RESULTS_DIR, "*_run*.csv")
files = sorted(glob.glob(pattern))

if not files:
    print(f"No trace CSV files found with pattern: {pattern}")
    raise SystemExit(0)

def load_trace_safe(path):
    """
    Load CSV, tolerate different column sets (some traces may have extra columns).
    Expect at least: iter, F, f, PenT, PenH, PenS, elapsed
    If columns missing, try to infer by name; if iter missing, create a 1..N index.
    """
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # ensure iter present
    if "iter" not in df.columns:
        df.insert(0, "iter", np.arange(1, len(df) + 1))
    # ensure f present
    if "f" not in df.columns and "travel" in df.columns:
        df = df.rename(columns={"travel": "f"})
    # if some penalty columns missing, create them with NaN so logic is robust
    for c in ("PenT", "PenH", "PenS"):
        if c not in df.columns:
            df[c] = np.nan
    # ensure columns are numeric where possible
    for c in ("iter", "F", "f", "PenT", "PenH", "PenS", "elapsed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_best_feasible_series(df):
    """
    For each iteration in df, compute the best feasible f observed so far.
    - We only update best when PenT == PenH == PenS == 0 (strict equality).
    - f must be > 0 to be considered valid.
    Returns a numpy array (same length as df) with np.nan where no feasible solution yet.
    """
    best_val = math.inf
    best_series = []
    for _, row in df.iterrows():
        f = row.get("f", None)
        PenT = row.get("PenT", None)
        PenH = row.get("PenH", None)
        PenS = row.get("PenS", None)

        # require exact zeros for all penalties to declare "feasible"
        is_feasible = (PenT == 0) and (PenH == 0) and (PenS == 0)
        if is_feasible and (f is not None) and (f > 0) and (not math.isnan(f)):
            if f < best_val:
                best_val = float(f)
        best_series.append(best_val if best_val < math.inf else np.nan)
    return np.array(best_series, dtype=float)

for path in files:
    try:
        df = load_trace_safe(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        continue

    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    print(f"Processing {base}  (rows={len(df)})")

    if "f" not in df.columns:
        print(" -> no 'f' column found, skipping.")
        continue

    iters = df["iter"].to_numpy()
    f_vals = df["f"].to_numpy()

    best_series = compute_best_feasible_series(df)

    # summary info
    # find last finite best feasible value
    finite_idxs = np.where(np.isfinite(best_series))[0]
    if finite_idxs.size > 0:
        last_idx = finite_idxs[-1]
        best_feasible_val = best_series[last_idx]
        best_feasible_iter = int(iters[last_idx])
        summary = f"best feasible f={best_feasible_val:.3f} at iter={best_feasible_iter}"
    else:
        best_feasible_val = None
        best_feasible_iter = None
        summary = "no feasible (PenT/PenH/PenS==0) iteration found"

    print("  ->", summary)

    # Plotting
    plt.figure(figsize=(9, 5))
    plt.title(f"{name}  — convergence (raw f in blue dots, best feasible in orange)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective f (travel distance)")

    # raw f as blue dots (no connecting line)
    plt.plot(iters, f_vals, linestyle="-", linewidth=2.0, label="raw f (per iter)", zorder=2, color="blue")

    # best feasible series as orange continuous line (no dots)
    if np.any(np.isfinite(best_series)):
        plt.plot(iters, best_series, linestyle="-", linewidth=3.0, label="best feasible f (so far)", zorder=3, color="orange")
    else:
        # still draw an empty legend entry so legend is consistent
        plt.plot([], [], linestyle="-", linewidth=2.0, label="best feasible f (so far)", color="orange")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{name}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> saved plot to {out_path}")

print("\nDone. Plots saved to:", PLOTS_DIR)
