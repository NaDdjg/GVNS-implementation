import os
import math
import numpy as np
import pandas as pd
from itertools import combinations


# ---------------------------
# Parsing
# ---------------------------

def read_instance(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() != ""]

    idx = 0
    m = int(lines[idx]); idx += 1
    n = int(lines[idx]); idx += 1
    H = int(lines[idx]); idx += 1
    n_sync = int(lines[idx]); idx += 1

    sync_ids = []
    if n_sync > 0:
        sync_ids = list(map(int, lines[idx].split()))
        idx += 1

    data = []
    for _ in range(n + 1):  # depot + customers
        parts = list(map(float, lines[idx].split()))
        idx += 1
        data.append(parts)

    df = pd.DataFrame(
        data,
        columns=["id", "x", "y", "a", "b", "service"]
    )

    return {
        "m": m,
        "n": n,
        "H": H,
        "sync_ids": set(sync_ids),
        "df": df
    }


# ---------------------------
# Statistics
# ---------------------------

def compute_tw_stats(df, H):
    customers = df[df["id"] != 0]
    widths = customers["b"] - customers["a"]
    rel_widths = widths / H

    return {
        "tw_mean": widths.mean(),
        "tw_std": widths.std(),
        "tw_min": widths.min(),
        "tw_max": widths.max(),
        "rel_tw_mean": rel_widths.mean(),
        "rel_tw_std": rel_widths.std()
    }


def compute_tw_overlap(df):
    customers = df[df["id"] != 0]
    intervals = list(zip(customers["a"], customers["b"]))

    overlaps = []
    for (a1, b1), (a2, b2) in combinations(intervals, 2):
        overlap = max(0, min(b1, b2) - max(a1, a2))
        overlaps.append(overlap > 0)

    overlaps = np.array(overlaps, dtype=int)

    if len(overlaps) == 0:
        return {
            "tw_overlap_ratio": 0.0,
            "avg_tw_degree": 0.0
        }

    n = len(customers)
    total_pairs = n * (n - 1) / 2

    return {
        "tw_overlap_ratio": overlaps.sum() / total_pairs,
        "avg_tw_degree": 2 * overlaps.sum() / n
    }


def compute_load_stats(df, m, H):
    customers = df[df["id"] != 0]
    total_service = customers["service"].sum()

    return {
        "total_service": total_service,
        "load_tightness": total_service / (m * H),
        "service_per_customer": customers["service"].mean()
    }


def compute_sync_stats(df, sync_ids):
    customers = df[df["id"] != 0]
    sync_customers = customers[customers["id"].isin(sync_ids)]

    return {
        "n_sync": len(sync_customers),
        "sync_ratio": len(sync_customers) / len(customers) if len(customers) > 0 else 0,
        "sync_service_ratio": (
            sync_customers["service"].sum() / customers["service"].sum()
            if customers["service"].sum() > 0 else 0
        )
    }


# ---------------------------
# Main analysis
# ---------------------------

def analyze_instance(path):
    inst = read_instance(path)
    df = inst["df"]

    stats = {
        "instance": os.path.basename(path),
        "m": inst["m"],
        "n": inst["n"],
        "H": inst["H"]
    }

    stats.update(compute_tw_stats(df, inst["H"]))
    stats.update(compute_tw_overlap(df))
    stats.update(compute_load_stats(df, inst["m"], inst["H"]))
    stats.update(compute_sync_stats(df, inst["sync_ids"]))

    return stats


def analyze_folder(folder):
    rows = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            rows.append(analyze_instance(path))

    return pd.DataFrame(rows)


# ---------------------------
# Run
# ---------------------------

if __name__ == "__main__":
    INSTANCES_DIR = "instances\\New_AB\\45"
    OUTPUT_CSV = "result_summaries\statistics\instance_45_AB_statistics.csv"

    df_stats = analyze_folder(INSTANCES_DIR)
    df_stats.to_csv(OUTPUT_CSV, index=False)

    print(df_stats)
    print("\nSaved to:", OUTPUT_CSV)