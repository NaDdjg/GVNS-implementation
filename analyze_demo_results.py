import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

BASE_RESULTS = Path("results/demo")

DISPLAY_NAMES = {
    "GVNS_PLUS": "GVNS+",
    "PPO": "PPO",
    "QLEARNING": "QL",
    "DQN": "DQN",
    "BF_SELECTOR": "BF",
    "QLEARNING_BF": "QL+BF",
    "DQN_BF": "DQN+BF",
}

INSTANCES = [
    "clustered_demo",
    "random_demo",
    "mixed_demo",
]

# ============================================================
# Utilities
# ============================================================

def load_trace_safe(path):
    """
    Robust trace loading.
    """

    df = pd.read_csv(path)

    # remove trailing empty rows
    df = df.dropna(how="all")

    df.columns = [c.strip() for c in df.columns]

    if "iter" not in df.columns:
        df.insert(0, "iter", np.arange(1, len(df) + 1))

    if "elapsed" not in df.columns:
        df["elapsed"] = np.arange(len(df))

    for c in ["iter", "f", "F", "PenT", "PenH", "PenS", "elapsed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_summary_safe(path):
    """
    Load summary CSV and remove trailing empty rows.
    """

    df = pd.read_csv(path)
    df = df.dropna(how="all")

    return df.iloc[0]


def feasible_mask(df):
    """
    Feasible = all penalties are zero.
    """

    return (
        (df["PenT"] == 0)
        & (df["PenH"] == 0)
        & (df["PenS"] == 0)
    )


def best_feasible_series(df):
    """
    Best feasible cost so far.
    """

    best = math.inf
    values = []

    for _, row in df.iterrows():

        feasible = (
            row["PenT"] == 0
            and row["PenH"] == 0
            and row["PenS"] == 0
        )

        if feasible and not pd.isna(row["f"]):
            best = min(best, float(row["f"]))

        values.append(best if best < math.inf else np.nan)

    return np.array(values)


# ============================================================
# Per-model plots
# ============================================================

def generate_model_plots():

    print("\nGenerating individual model plots...")

    for model_folder in DISPLAY_NAMES.keys():

        model_path = BASE_RESULTS / model_folder

        if not model_path.exists():
            continue

        analysis_dir = model_path / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        run_files = sorted(model_path.glob("*_run*.csv"))

        for trace_file in run_files:

            df = load_trace_safe(trace_file)

            instance_name = trace_file.stem.replace("_run1", "")

            best_series = best_feasible_series(df)

            # ====================================================
            # TIME PLOT
            # ====================================================

            plt.figure(figsize=(10, 5))

            plt.plot(
                df["elapsed"],
                df["f"],
                linewidth=1.5,
                label="Current cost"
            )

            plt.step(
                df["elapsed"],
                best_series,
                where="post",
                linewidth=2.5,
                label="Best feasible cost"
            )

            plt.xlabel("Time (s)")
            plt.ylabel("Cost")
            plt.title(
                f"{DISPLAY_NAMES[model_folder]} - {instance_name}"
            )

            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                analysis_dir / f"{instance_name}_time.png",
                dpi=300
            )

            plt.close()

            # ====================================================
            # ITERATION PLOT
            # ====================================================

            plt.figure(figsize=(10, 5))

            plt.plot(
                df["iter"],
                df["f"],
                linewidth=1.5,
                label="Current cost"
            )

            plt.step(
                df["iter"],
                best_series,
                where="post",
                linewidth=2.5,
                label="Best feasible cost"
            )

            plt.xlabel("Iteration")
            plt.ylabel("Cost")

            plt.title(
                f"{DISPLAY_NAMES[model_folder]} - {instance_name}"
            )

            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                analysis_dir / f"{instance_name}_iterations.png",
                dpi=300
            )

            plt.close()

    print("Done.")


# ============================================================
# Tables
# ============================================================

def generate_tables():

    print("\nGenerating tables...")

    for instance in INSTANCES:

        results_rows = []
        feasibility_rows = []
        improvement_rows = []

        for model_folder, model_name in DISPLAY_NAMES.items():

            model_path = BASE_RESULTS / model_folder

            summary_file = (
                model_path /
                f"{instance}_summary.csv"
            )

            trace_file = (
                model_path /
                f"{instance}_run1.csv"
            )

            if (
                not summary_file.exists()
                or
                not trace_file.exists()
            ):
                continue

            summary = load_summary_safe(summary_file)
            trace = load_trace_safe(trace_file)

            # ====================================================
            # RESULTS TABLE
            # ====================================================

            best_cost = float(summary["best_f"])
            best_time = float(summary["time"])

            results_rows.append({
                "Model": model_name,
                "Best Cost": best_cost,
                "Time To Best (s)": best_time
            })

            # ====================================================
            # FEASIBILITY TABLE
            # ====================================================

            feasible = feasible_mask(trace)

            feasible_count = int(feasible.sum())
            total_count = len(trace)

            ratio = (
                100 * feasible_count / total_count
            )

            feasibility_rows.append({
                "Model": model_name,
                "Feasible Iterations": feasible_count,
                "Total Iterations": total_count,
                "Feasibility (%)": ratio
            })

            # ====================================================
            # IMPROVEMENT TABLE
            # ====================================================

            initial_cost = float(trace.iloc[0]["f"])

            improvement = (
                100.0
                * (initial_cost - best_cost)
                / initial_cost
            )

            improvement_rows.append({
                "Model": model_name,
                "Initial Cost": initial_cost,
                "Best Cost": best_cost,
                "Improvement (%)": improvement
            })

        pd.DataFrame(results_rows).sort_values(
            "Best Cost"
        ).to_csv(
            BASE_RESULTS /
            f"{instance}_results_table.csv",
            index=False
        )

        pd.DataFrame(feasibility_rows).sort_values(
            "Feasibility (%)",
            ascending=False
        ).to_csv(
            BASE_RESULTS /
            f"{instance}_feasibility_table.csv",
            index=False
        )

        pd.DataFrame(improvement_rows).sort_values(
            "Improvement (%)",
            ascending=False
        ).to_csv(
            BASE_RESULTS /
            f"{instance}_improvement_table.csv",
            index=False
        )

    print("Done.")


# ============================================================
# Comparison plots
# ============================================================

def generate_comparison_plots():

    print("\nGenerating comparison plots...")

    for instance in INSTANCES:

        plt.figure(figsize=(11, 6))

        for model_folder, model_name in DISPLAY_NAMES.items():

            trace_file = (
                BASE_RESULTS
                / model_folder
                / f"{instance}_run1.csv"
            )

            if not trace_file.exists():
                continue

            trace = load_trace_safe(trace_file)

            best_series = best_feasible_series(trace)

            plt.step(
                trace["elapsed"],
                best_series,
                where="post",
                linewidth=2.5,
                label=model_name
            )

        plt.xlabel("Time (s)")
        plt.ylabel("Best feasible cost")
        plt.title(
            f"Model Comparison - {instance}"
        )

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            BASE_RESULTS /
            f"comparison_{instance}.png",
            dpi=300
        )

        plt.close()

    print("Done.")


# ============================================================
# Time-to-best ranking plots
# ============================================================

def generate_time_to_best_plots():

    print("\nGenerating ranking plots...")

    for instance in INSTANCES:

        rows = []

        for model_folder, model_name in DISPLAY_NAMES.items():

            summary_file = (
                BASE_RESULTS
                / model_folder
                / f"{instance}_summary.csv"
            )

            if not summary_file.exists():
                continue

            summary = load_summary_safe(summary_file)

            rows.append({
                "Model": model_name,
                "Time": float(summary["time"])
            })

        if not rows:
            continue

        df = pd.DataFrame(rows)

        df = df.sort_values("Time")

        plt.figure(figsize=(8, 5))

        plt.barh(
            df["Model"],
            df["Time"]
        )

        plt.xlabel("Time to best solution (s)")

        plt.title(
            f"Time To Best - {instance}"
        )

        plt.tight_layout()

        plt.savefig(
            BASE_RESULTS /
            f"time_to_best_{instance}.png",
            dpi=300
        )

        plt.close()

    print("Done.")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("DEMO RESULTS ANALYSIS")
    print("=" * 60)

    generate_model_plots()

    generate_tables()

    generate_comparison_plots()

    generate_time_to_best_plots()

    print("\nFinished.")
    print(f"Results saved in: {BASE_RESULTS}")


if __name__ == "__main__":
    main()