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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a neighborhood usage pie chart from a CSV file."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the neighborhood usage CSV file.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder where the PNG file will be saved.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output filename for the PNG (default derives from the input CSV name).",
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

    if not csv_path.exists():
        print(f"Error: input file not found: {csv_path}", file=sys.stderr)
        return 1

    output_folder.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{csv_path.stem}_neighborhood_usage.png"
    output_path = output_folder / output_name
    instance_name = extract_instance_name(csv_path)

    try:
        counts = load_usage_counts(csv_path)
        create_pie_chart(counts, output_path, instance_name)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Pie chart saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
