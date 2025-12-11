import os
from gvns import benchmark

# rl utilities
from gvns.agents.qlearning_agent import QTableAgent
from gvns.utils import NEIGHBORHOOD_NAMES

# parameters
N_RUNS = 1  
INST_FOLDER = "instances"      
# RESULTS_FOLDER = "results-classical"
# RESULTS_FOLDER = "results-qlearning"
# RESULTS_FOLDER = "results-Fist-4"
RESULTS_FOLDER = "results-Best-4"

# GVNS hyperparameters 
params = {
    "Contmax": 100,
    "p_init": 1,
    "p_max": 10,
    "Dp": 0.3,
    "w1": 0.8,    # kan 0.5, mais to have less time in the start to solve penalties
    "w2": 0.1,
    "alpha": 0.1, # kan 0.1 mais to make penalties solved in a stronger way
    "delta": 0.4, # kan 0.4 mais to keep accpetance tae worse f meshi kbira 
    "M": 1e6,
    "seed": 42,
    "Tmax": 200  
}

# ----------------------------
# Agent configuration
# ----------------------------
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

# run benchmark
if __name__ == "__main__":
    benchmark.benchmark_main(
        INST_FOLDER=INST_FOLDER,
        RESULTS_FOLDER=RESULTS_FOLDER,
        PARAMS=params,
        agent=agent,
        N_RUNS=N_RUNS
    )