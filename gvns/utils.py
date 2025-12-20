import numpy as np, time
import csv
import os
import matplotlib.pyplot as plt
from run_benchmark import RESULTS_FOLDER
from gvns.benchmark import NEIGHBORHOOD_FUNCS

NEIGHBORHOOD_NAMES = ['N1','N2','N3','N4','N5','N6','N7','N8']
NEIGHBORHOODS = NEIGHBORHOOD_NAMES

# ===================================================================
# local search utils 
# ===================================================================
def save_neighborhood_usage(counter, instance_name):
    """
    Saves counts of usage of each neighborhood during the local search
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

def save_neighborhood_heatmap(counter, instance_name):
    """
    Creates heatmaps for the usage of each neighborhood 
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
    """
    Crates an order suggestion based on the usage counts of each neighborhood
    """
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
    Used when the local search mode is "suggested"
    Reads the suggestion file and return a list of neighborhood indices (0-based).
    lines are like these :  N1 — usage = X
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

# ===================================================================
# rl utils
# ===================================================================
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = 1.0 if x.ndim == 1 else x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        # Welford-style online update of mean and variance
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)

def extract_state(inst, current_sol, iter_no, start_time=None, neigh_success=None, neigh_attempts=None):
    n = inst.n
    tw_widths = [
        node["due"] - node["ready"]
        for node in inst.nodes.values()
    ]
    avg_tw = float(np.mean(tw_widths)) if len(tw_widths) > 0 else 0.0
    sync_ratio = len(getattr(inst, "sync_pairs", [])) / max(1, n)

    feats = [float(n), avg_tw, sync_ratio]

    evalf = [
        float(current_sol.Fval),
        float(current_sol.travel_time),
        float(current_sol.penT),
        float(current_sol.penH),
        float(current_sol.penS),
    ]

    progress = [float(iter_no), float(time.time() - (start_time or 0.0))]

    succ_rates = []
    if neigh_success is not None and neigh_attempts is not None:
        for s, a in zip(neigh_success, neigh_attempts):
            succ_rates.append(s / a if a > 0 else 0.0)
    else:
        succ_rates = [0.0] * len(NEIGHBORHOODS)

    succ_rates = [float(np.clip(x, 0.0, 1.0)) for x in succ_rates]
    
    return np.array(feats + evalf + progress + succ_rates, dtype=np.float32)

def obs_dim(n_neigh: int = None) -> int:
    """return observation vector dimension: (3 instance feats) + (5 eval) + (2 progress) + n_neigh"""
    if n_neigh is None:
        n_neigh = len(NEIGHBORHOODS)
    return 3 + 5 + 2 + n_neigh