"""
===========================================================
VRPTW-S Instance Visualization Script
===========================================================

This script generates TWO publication-quality figures:

1. Spatial Graph
   - Depot
   - Customers
   - Synchronized customers highlighted
   - Customer IDs displayed

2. Time Window Gantt Plot
   - Each customer's time window
   - Synchronized customers highlighted

-----------------------------------------------------------
Expected instance format
-----------------------------------------------------------

25
50
1236
5
21 5 7 26 3
0 40 50 0 1236 0
1 45 68 0 1127 90
...

Metadata:
1. Number of vehicles
2. Number of customers
3. Horizon
4. Number of synchronized customers
5. IDs of synchronized customers

Then:
id x y ready_time due_time service_time

-----------------------------------------------------------
Usage
-----------------------------------------------------------

python visualize_instance.py path/to/instance.txt

Example:
python visualize_instance.py C106_100_5.txt

-----------------------------------------------------------
Outputs
-----------------------------------------------------------

- spatial_graph.png
- spatial_graph.pdf
- time_windows.png
- time_windows.pdf

===========================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Input instance file
INPUT_FILE = r"instances/New_synch/50/RC202_50_5.txt"

# Output directory
OUTPUT_DIR = r"result_summaries/plots/instance_representation"

# =========================================================
# PARSER
# =========================================================

def parse_instance(file_path):
    """
    Parse VRPTW-S instance file.
    """

    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    num_vehicles = int(lines[0])
    num_customers = int(lines[1])
    horizon = int(lines[2])

    num_sync = int(lines[3])

    sync_customers = list(map(int, lines[4].split()))

    # -----------------------------------------------------
    # Customer data
    # -----------------------------------------------------

    data = []

    for line in lines[5:]:
        parts = line.split()

        customer = {
            "id": int(parts[0]),
            "x": float(parts[1]),
            "y": float(parts[2]),
            "ready_time": float(parts[3]),
            "due_time": float(parts[4]),
            "service_time": float(parts[5]),
        }

        data.append(customer)

    df = pd.DataFrame(data)

    return {
        "num_vehicles": num_vehicles,
        "num_customers": num_customers,
        "horizon": horizon,
        "sync_customers": sync_customers,
        "data": df,
    }


# =========================================================
# SPATIAL GRAPH
# =========================================================

def plot_spatial_graph(instance, save_dir="."):
    """
    Create spatial representation of the instance.
    """

    df = instance["data"]
    sync_customers = instance["sync_customers"]

    # -----------------------------------------------------
    # Separate node types
    # -----------------------------------------------------

    depot = df[df["id"] == 0]

    normal = df[
        (df["id"] != 0) &
        (~df["id"].isin(sync_customers))
    ]

    sync = df[df["id"].isin(sync_customers)]

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    plt.figure(figsize=(12, 10))

    # -----------------------------------------------------
    # Plot normal customers
    # -----------------------------------------------------

    plt.scatter(
        normal["x"],
        normal["y"],
        s=80,
        marker="o",
        alpha=0.8,
        label="Customers"
    )

    # -----------------------------------------------------
    # Plot synchronized customers
    # -----------------------------------------------------

    plt.scatter(
        sync["x"],
        sync["y"],
        s=200,
        marker="*",
        edgecolors="gray",
        linewidths=1.2,
        label="Synchronized Customers"
    )

    # -----------------------------------------------------
    # Plot depot
    # -----------------------------------------------------

    plt.scatter(
        depot["x"],
        depot["y"],
        s=200,
        marker="s",
        edgecolors="gray",
        linewidths=1.5,
        label="Depot"
    )

    # -----------------------------------------------------
    # Annotate IDs
    # -----------------------------------------------------

    for _, row in df.iterrows():

        x = row["x"]
        y = row["y"]

        cid = int(row["id"])

        ready = int(row["ready_time"])
        due = int(row["due_time"])

        # Depot label
        if cid == 0:

            label = f"DEPOT\n[{ready}, {due}]"

            fontsize = 9
            fontweight = "bold"

        else:

            label = f"{cid}\n[{ready}, {due}]"

            fontsize = 7
            fontweight = "normal"

        plt.text(
            x + 0.8,
            y + 0.8,
            label,
            fontsize=fontsize,
            fontweight=fontweight,
        )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    plt.title(
        "VRPTW-S Spatial Instance Representation",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.axis("equal")

    plt.legend()

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    png_path = os.path.join(save_dir, "spatial_graph.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved: {png_path}")


# =========================================================
# TIME WINDOW GANTT PLOT
# =========================================================

def plot_time_windows(instance, save_dir="."):
    """
    Create Gantt-style time window plot.
    """

    df = instance["data"].copy()
    sync_customers = instance["sync_customers"]

    # -----------------------------------------------------
    # Sort by ready time
    # -----------------------------------------------------

    df = df.sort_values(by="ready_time")

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    plt.figure(figsize=(16, 12))

    # -----------------------------------------------------
    # Plot each customer TW
    # -----------------------------------------------------

    for idx, (_, row) in enumerate(df.iterrows()):

        cid = int(row["id"])

        start = row["ready_time"]
        end = row["due_time"]

        width = end - start

        # synchronized customer
        if cid in sync_customers:
            alpha = 1.0
            edgecolor = "black"
            linewidth = 1.5

        else:
            alpha = 0.7
            edgecolor = "none"
            linewidth = 0

        plt.barh(
            y=idx,
            width=width,
            left=start,
            height=0.7,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth
        )

        # Label with customer ID
        plt.text(
            start - 15,
            idx,
            str(cid),
            va="center",
            fontsize=8
        )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    plt.title(
        "Customer Time Windows (Gantt Representation)",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Time")
    plt.ylabel("Customers")

    plt.grid(True, axis="x", linestyle="--", alpha=0.4)

    # remove useless y-axis ticks
    plt.yticks([])

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------

    legend_elements = [
        Line2D(
            [0],
            [0],
            lw=10,
            label="Normal Customer"
        ),
        Line2D(
            [0],
            [0],
            lw=10,
            marker='s',
            markersize=10,
            markeredgecolor='black',
            label="Synchronized Customer"
        )
    ]

    plt.legend(handles=legend_elements)

    plt.tight_layout()

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    png_path = os.path.join(save_dir, "time_windows.png")
    pdf_path = os.path.join(save_dir, "time_windows.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    plt.close()

    print(f"[INFO] Saved: {png_path}")
    print(f"[INFO] Saved: {pdf_path}")


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Create output directory if it doesn't exist
    # -----------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------
    # Parse instance
    # -----------------------------------------------------

    instance = parse_instance(INPUT_FILE)

    print("\n================================================")
    print("INSTANCE SUMMARY")
    print("================================================")

    print(f"Vehicles               : {instance['num_vehicles']}")
    print(f"Customers              : {instance['num_customers']}")
    print(f"Horizon                : {instance['horizon']}")
    print(f"Synchronized Customers : {instance['sync_customers']}")

    print("================================================\n")

    # -----------------------------------------------------
    # Generate figures
    # -----------------------------------------------------

    plot_spatial_graph(instance, OUTPUT_DIR)

    print("\n[INFO] Visualization completed successfully.\n")


if __name__ == "__main__":
    main()