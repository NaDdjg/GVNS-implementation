import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

NEIGHBORHOOD_LABELS = {
    0: "intra_insertion",
    1: "inter_insertion",
    2: "intra_swap",
    3: "inter_swap",
    4: "two_opt",
    5: "two_opt_star",
    6: "cross_exchange",
    7: "k_relocate",
}


def load_usage_counts(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if "Usage_Count" not in df.columns and "Usage_count" not in df.columns:
        raise ValueError(
            f"CSV file must contain a 'Usage_Count' column, found: {list(df.columns)}"
        )

    # Support either exact or lowercase variants
    if "Usage_Count" in df.columns:
        usage_col = "Usage_Count"
    else:
        usage_col = "Usage_count"

    if "Neighborhood_ID" not in df.columns and "Neighborhood_Id" not in df.columns and "neighborhood_id" not in df.columns:
        raise ValueError(
            f"CSV file must contain a 'Neighborhood_ID' column, found: {list(df.columns)}"
        )

    if "Neighborhood_ID" in df.columns:
        id_col = "Neighborhood_ID"
    elif "Neighborhood_Id" in df.columns:
        id_col = "Neighborhood_Id"
    else:
        id_col = "neighborhood_id"

    counts = df.set_index(id_col)[usage_col]
    counts = counts.groupby(level=0).sum().reindex(sorted(NEIGHBORHOOD_LABELS.keys()), fill_value=0)
    return counts


def create_pie_chart(counts: pd.Series, output_path: Path, instance_name: str) -> None:
    labels = []
    sizes = []

    for neigh_id, count in counts.items():
        label = NEIGHBORHOOD_LABELS.get(neigh_id, f"neigh_{neigh_id}")
        labels.append(label)
        sizes.append(count)

    if sum(sizes) == 0:
        raise ValueError("No usage counts found in the file. The pie chart cannot be generated from all-zero values.")

    colors = [
        "#88CCEE",
        "#CC6677",
        "#DDCC77",
        "#117733",
        "#332288",
        "#AA4499",
        "#44AA99",
        "#999933",
    ]

    def autopct_func(pct):
        return f"{pct:.1f}%" if pct > 0 else ""

    fig, (chart_ax, legend_ax) = plt.subplots(
        ncols=2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [2, 1], "wspace": 0.002},
    )
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08)

    wedges, texts, autotexts = chart_ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=autopct_func,
        startangle=90,
        pctdistance=0.75,
        textprops={"color": "black", "fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    chart_ax.axis("equal")
    chart_ax.set_xticks([])
    chart_ax.set_yticks([])

    legend_ax.axis("off")
    legend_labels = [f"{label}: {count:,}" for label, count in zip(labels, sizes)]
    legend = legend_ax.legend(
        wedges,
        legend_labels,
        title="Neighborhood (usage count)",
        loc="center",
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc",
        fontsize=10,
        title_fontsize=11,
    )
    legend.get_frame().set_boxstyle("round,pad=0.5")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def load_iteration_trace(csv_path: Path) -> pd.DataFrame:
    """Load iteration-level trace data with neighborhood selection and improvements."""
    df = pd.read_csv(csv_path)

    # Normalize column names to lowercase for flexible matching
    df.columns = [col.lower() for col in df.columns]

    # Validate required columns (lowercase)
    required_cols = {"iteration", "neighborhood_id", "f_before", "f_after"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns: {required_cols}, found: {set(df.columns)}"
        )

    # Coerce numeric columns and drop rows with missing essential values
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["neighborhood_id"] = pd.to_numeric(df["neighborhood_id"], errors="coerce")
    df["f_before"] = pd.to_numeric(df["f_before"], errors="coerce")
    df["f_after"] = pd.to_numeric(df["f_after"], errors="coerce")

    df = df.dropna(subset=["iteration", "neighborhood_id", "f_before", "f_after"]) 

    # Ensure neighborhood ids are integers
    try:
        df["neighborhood_id"] = df["neighborhood_id"].astype(int)
    except Exception:
        df["neighborhood_id"] = df["neighborhood_id"].round().astype(int)

    # Calculate improvement
    df["improvement"] = df["f_before"] - df["f_after"]

    return df


def create_timeseries_plots(
    trace_path: Path, output_folder: Path, instance_name: str
) -> None:
    """Create multi-panel time-series visualization of neighborhood usage."""
    df = load_iteration_trace(trace_path)
    
    # Define colors for neighborhoods
    colors_map = {
        0: "#88CCEE",
        1: "#CC6677",
        2: "#DDCC77",
        3: "#117733",
        4: "#332288",
        5: "#AA4499",
        6: "#44AA99",
        7: "#999933",
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 11))
    
    # --- Panel 1: Neighborhood Selection Timeline ---
    ax1 = axes[0]
    df_sorted = df.sort_values("iteration")
    
    # Plot each neighborhood as a colored line
    for neigh_id in sorted(df["neighborhood_id"].unique()):
        neigh_data = df_sorted[df_sorted["neighborhood_id"] == neigh_id]
        label = NEIGHBORHOOD_LABELS.get(neigh_id, f"N{neigh_id}")
        ax1.scatter(
            neigh_data["iteration"],
            [neigh_id] * len(neigh_data),
            label=label,
            color=colors_map.get(neigh_id, "#CCCCCC"),
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
    
    # Draw connecting line to show sequence
    ax1.plot(
        df_sorted["iteration"],
        df_sorted["neighborhood_id"],
        color="#AAAAAA",
        alpha=0.3,
        linewidth=1,
        zorder=0,
    )
    
    ax1.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Neighborhood ID", fontsize=11, fontweight="bold")
    ax1.set_title("Neighborhood Selection Sequence", fontsize=12, fontweight="bold")
    ax1.set_yticks(sorted(df["neighborhood_id"].unique()))
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(loc="upper center", ncol=4, fontsize=9, framealpha=0.95)
    
    # --- Panel 2: Cumulative Usage Over Time ---
    ax2 = axes[1]
    
    for neigh_id in sorted(NEIGHBORHOOD_LABELS.keys()):
        if neigh_id in df["neighborhood_id"].values:
            neigh_data = df_sorted[df_sorted["neighborhood_id"] == neigh_id]
            cumulative = (df_sorted["neighborhood_id"] == neigh_id).cumsum()
            label = NEIGHBORHOOD_LABELS.get(neigh_id, f"N{neigh_id}")
            ax2.plot(
                df_sorted["iteration"],
                cumulative,
                label=label,
                color=colors_map.get(neigh_id, "#CCCCCC"),
                linewidth=2.5,
                marker="o",
                markersize=3,
                alpha=0.8,
            )
    
    ax2.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Usage Count", fontsize=11, fontweight="bold")
    ax2.set_title("Cumulative Neighborhood Usage Over Time", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)
    
    
    fig.tight_layout()
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{instance_name}_timeseries.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Time-series plot saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate neighborhood usage visualization from CSV files."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the neighborhood usage CSV file (with Usage_Count).",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder where the PNG files will be saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename for the PNG (default derives from the input CSV name).",
    )
    parser.add_argument(
        "--trace-csv",
        type=Path,
        default=None,
        help="Optional path to iteration trace CSV file (with iteration, neighborhood_id, F_before, F_after) to generate time-series plots.",
    )
    return parser.parse_args()


def extract_instance_name(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.endswith("_usage"):
        return stem[: -len("_usage")]
    if stem.endswith("_Usage"):
        return stem[: -len("_Usage")]
    return stem


def main() -> int:
    args = parse_args()
    csv_path = args.input_csv
    output_folder = args.output_folder
    trace_csv = args.trace_csv

    if not csv_path.exists():
        print(f"Error: input file not found: {csv_path}", file=sys.stderr)
        return 1

    output_folder.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{csv_path.stem}_neighborhood_usage.png"
    output_path = output_folder / output_name
    instance_name = extract_instance_name(csv_path)

    try:
        # Generate pie chart from usage counts
        counts = load_usage_counts(csv_path)
        create_pie_chart(counts, output_path, instance_name)
        
        # Generate time-series plots if trace CSV is provided
        if trace_csv:
            if not trace_csv.exists():
                print(f"Error: trace file not found: {trace_csv}", file=sys.stderr)
                return 1
            create_timeseries_plots(trace_csv, output_folder, instance_name)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Pie chart saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
