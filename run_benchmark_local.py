import os
import warnings
warnings.filterwarnings('ignore')

from gvns import benchmark

# rl utilities
from gvns.agents.qlearning_agent import QTableAgent

# parameters
N_RUNS = 1  
NEIGHBORHOOD_NAMES = ['N1','N2','N3','N4','N5','N6','N7','N8']

# ============================================================
# saving paths
# ============================================================
INST_FOLDER = "instances/30"      
RESULTS_FOLDER = "results\\without RL\\predefined_order\\Best"
# RESULTS_FOLDER = "results\\without RL\\predefined_order\\First"
# RESULTS_FOLDER = "results\\without RL\\random_order\\Best"
# RESULTS_FOLDER = "results\\without RL\\random_order\\First"
# RESULTS_FOLDER = "results\\without RL\\reverse_order\\Best"
# RESULTS_FOLDER = "results\\without RL\\reverse_order\\First"
# RESULTS_FOLDER = "results\\without RL\\suggested_order\\Best"
# RESULTS_FOLDER = "results\\without RL\\suggested_order\\First"
# RESULTS_FOLDER = "results\\with RL\\qlearning\\Best"
# RESULTS_FOLDER = "results\\with RL\\qlearning\\First"

# ============================================================
# GVNS hyperparameters 
# neigh_order can be: "default", "reverse", "suggested"
# ============================================================
params = {
    "log_path": "result_summaries/debug/gvns_trace.jsonl",
    "Contmax": 100,
    "p_init": 1,
    "p_max": 10,
    "Dp": 0.3,
    "w1": 0.5,    
    "w2": 0.1,
    "alpha": 0.1, 
    "delta": 0.4, 
    "M": 1e6,
    "seed": 42,
    "Tmax": 500,
    "neigh_order": "default"
}

# ============================================================
# Agent configuration
# ============================================================
n_actions = len(NEIGHBORHOOD_NAMES)
agent = QTableAgent(
    n_actions=n_actions,
    alpha=0.2,
    gamma=0.95,
    eps_start=0.1,
    eps_min=0.01,
    eps_decay=5e-4,
    use_ucb=True,
    seed=params["seed"]
)

# this is just a utility to ensure results folder exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

QTABLE_DIR = None
if agent is not None:
    QTABLE_DIR = os.path.join(RESULTS_FOLDER, "qtables")
    os.makedirs(QTABLE_DIR, exist_ok=True)

# run benchmark
if __name__ == "__main__":
    benchmark.benchmark_main(
        INST_FOLDER=INST_FOLDER,
        RESULTS_FOLDER=RESULTS_FOLDER,
        QTABLE_DIR=QTABLE_DIR,
        PARAMS=params,
        agent=None,
        first_improvement = True,
        N_RUNS=N_RUNS
    )