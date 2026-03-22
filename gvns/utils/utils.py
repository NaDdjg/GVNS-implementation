import csv
import os
import matplotlib.pyplot as plt
from run_benchmark import RESULTS_FOLDER
from gvns.local_search import NEIGHBORHOOD_FUNCS

NEIGHBORHOOD_NAMES = ['N1','N2','N3','N4','N5','N6','N7','N8']
NEIGHBORHOODS = NEIGHBORHOOD_NAMES

# ===================================================================
# local search utils 
# ===================================================================
def save_neighborhood_usage(counter, instance_name):
    """                                                                                                                                                                                                            
    saves counts of usage of each neighborhood during the local search
    """
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    neigh_folder = os.path.join(RESULTS_FOLDER, "neigh_usage")
    os.makedirs(neigh_folder, exist_ok=True)
    path = os.path.join(neigh_folder, f"{instance_name}_usage.csv")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Neighborhood_ID", "Usage_Count"])
        for i, count in enumerate(counter):
            w.writerow([i, count])

def save_neighborhood_improvement_log(events, instance_name, suffix="bvnd"):
    if not events:
        return

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    neigh_folder = os.path.join(RESULTS_FOLDER, "neigh_usage")
    os.makedirs(neigh_folder, exist_ok=True)

    path = os.path.join(
        neigh_folder,
        f"{instance_name}_neigh_improvements_{suffix}.csv"
    )

    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        # header from dict keys
        header = events[0].keys()
        w.writerow(header)

        for ev in events:
            w.writerow(ev.values())

def save_neighborhood_heatmap(counter, instance_name):
    """
    creates heatmaps for the usage of each neighborhood 
    """
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    neigh_folder = os.path.join(RESULTS_FOLDER, "neigh_usage")
    os.makedirs(neigh_folder, exist_ok=True)
    path = os.path.join(neigh_folder, f"{instance_name}_heatmap.png")

    plt.figure(figsize=(8, 2))
    plt.imshow([counter], cmap="viridis", aspect="auto")
    plt.colorbar(label="Usage Count")
    plt.yticks([])
    plt.xticks(range(len(counter)), [f"N{i+1}" for i in range(len(counter))])
    plt.title(f"Neighborhood Usage Heatmap - {instance_name}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def save_suggested_order(counter, instance_name, neighborhoods):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    neigh_folder = os.path.join(RESULTS_FOLDER, "neigh_usage")
    os.makedirs(neigh_folder, exist_ok=True)
    path = os.path.join(neigh_folder, f"{instance_name}_order_suggestion.txt")

    # pair = (neigh_id, usage)
    pairs = list(enumerate(counter))
    pairs.sort(key=lambda x: -x[1])

    with open(path, "w") as f:
        f.write(f"Suggested order for instance {instance_name}:\n")
        f.write("From most useful to least useful:\n\n")
        for neigh_id, usage in pairs:
            f.write(f"N{neigh_id+1} — usage = {usage}\n")

def load_suggested_order(instance_name):
    """
    used when the local search mode is "suggested"
    reads the suggestion file and return a list of neighborhood indices (0-based).
    lines are like these :  N1 — usage = X
    default order should be run before it because it takes the order from the saved files after the default order run.
    """
    path = os.path.join(
        RESULTS_FOLDER,
        "neigh_usage",
        f"{instance_name}_order_suggestion.txt"
    )

    if not os.path.exists(path):
        print(f"[WARNING] Suggested order file not found: {path}")
        print("Using default neighborhood order.")
        return list(range(len(NEIGHBORHOOD_FUNCS)))

    order = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("N"):
                try:
                    nid = int(line.split("—")[0].replace("N", "")) - 1
                    order.append(nid)
                except ValueError:
                    pass

    if len(order) != len(NEIGHBORHOOD_FUNCS):
        print("[WARNING] Suggested order incomplete — falling back to default order.")
        return list(range(len(NEIGHBORHOOD_FUNCS)))

    print(f"[INFO] Loaded suggested neighborhood order from {path}")
    print(f"[INFO] suggested neighborhood order {order}")
    return order


def save_operator_stats(operator_stats, instance_name, suffix="dqn"):

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    stats_folder = os.path.join(RESULTS_FOLDER, "operator_stats")
    os.makedirs(stats_folder, exist_ok=True)

    path = os.path.join(stats_folder, f"{instance_name}_{suffix}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "operator_id",
            "uses",
            "success",
            "success_rate",
            "avg_reward"
        ])

        for op, stats in operator_stats.items():
            writer.writerow([
                op,
                stats["uses"],
                stats["success"],
                stats["success_rate"],
                stats["avg_reward"]
            ])