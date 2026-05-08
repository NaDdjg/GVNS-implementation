import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results\methods\Simple_GVNS_F\plot_file"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_time")
os.makedirs(PLOTS_DIR, exist_ok=True)

pattern = os.path.join(RESULTS_DIR, "*_run*.csv")
files = sorted(glob.glob(pattern))

if not files:
    print(f"No trace CSV files found with pattern: {pattern}")
    raise SystemExit(0)


def load_trace_safe(path):
    """Load a GVNS trace CSV robustly and normalize its columns."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Fill missing essentials
    if "iter" not in df.columns:
        df.insert(0, "iter", np.arange(1, len(df) + 1))
    if "f" not in df.columns and "travel" in df.columns:
        df = df.rename(columns={"travel": "f"})
    if "elapsed" not in df.columns:
        # fallback: fake elapsed = index if missing
        df["elapsed"] = np.arange(len(df))
    for c in ("PenT", "PenH", "PenS"):
        if c not in df.columns:
            df[c] = np.nan
    for c in ("iter", "F", "f", "PenT", "PenH", "PenS", "elapsed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_best_feasible_series(df):
    """Return cumulative best feasible f (PenT==PenH==PenS==0) over iterations."""
    best_val = math.inf
    best_series = []
    for _, row in df.iterrows():
        f = row.get("f", None)
        PenT, PenH, PenS = row.get("PenT", None), row.get("PenH", None), row.get("PenS", None)
        feasible = (PenT == 0) and (PenH == 0) and (PenS == 0)
        if feasible and (f is not None) and (f > 0) and (not math.isnan(f)):
            best_val = min(best_val, float(f))
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
    print(f"Processing {base} (rows={len(df)})")

    if "f" not in df.columns or "elapsed" not in df.columns:
        print(" -> missing f or elapsed column, skipping.")
        continue

    time_vals = df["elapsed"].to_numpy()
    f_vals = df["f"].to_numpy()
    best_series = compute_best_feasible_series(df)

    # Summary
    finite_idxs = np.where(np.isfinite(best_series))[0]
    if finite_idxs.size > 0:
        last_idx = finite_idxs[-1]
        best_feasible_val = best_series[last_idx]
        best_feasible_time = time_vals[last_idx]
        summary = f"best feasible f={best_feasible_val:.3f} at t={best_feasible_time:.2f}s"
    else:
        summary = "no fully feasible solution (PenT/PenH/PenS==0)"
    print("  ->", summary)

    # ---- Plot ----
    plt.figure(figsize=(9, 5))
    plt.title(f"{name} — convergence vs time")
    plt.xlabel("Elapsed time (s)")
    plt.ylabel("Objective f (travel distance)")

    # Raw f curve with dots
    plt.plot(time_vals, f_vals, "-o", markersize=2, linewidth=1.5,
             label="raw f (current value)", color="blue", alpha=0.8, zorder=2)

    # Best feasible continuous orange line
    if np.any(np.isfinite(best_series)):
        plt.plot(time_vals, best_series, "-", linewidth=2.5,
                 label="best feasible f", color="orange", zorder=3)
    else:
        plt.plot([], [], "-", linewidth=2.5, label="best feasible f (so far)", color="orange")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(PLOTS_DIR, f"{name}_time.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  -> saved plot to {out_path}")

print("\nAll plots saved in:", PLOTS_DIR)
