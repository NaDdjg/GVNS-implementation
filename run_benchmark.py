import os
import argparse
import warnings
warnings.filterwarnings('ignore')

from gvns import benchmark

# rl utilities
from gvns.agents.qlearning_agent import QTableAgent
from gvns.utils import NEIGHBORHOOD_NAMES

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["first", "best", "rl"],
    help="Execution mode"
)

parser.add_argument(
    "--neigh_order",
    type=str,
    default="default",
    choices=["default", "reverse", "random", "suggested"],
    help="Neighborhood order (non-RL only)"
)

parser.add_argument(
    "--results_dir",
    type=str,
    default=None,
    help="optional custom results directory"
)

args = parser.parse_args()

# ============================================================
# GVNS hyperparameters 
# neigh_order can be: "default", "reverse", "random", "suggested"
# ============================================================
N_RUNS = 1  
INST_FOLDER = "instances"   

params = {
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
# Mode-dependent configuration
# ============================================================
agent = None
first_improvement = False

#  Fisrst improvement GVNS
if args.mode == "first":
    first_improvement = True
    params["neigh_order"] = args.neigh_order

    RESULTS_FOLDER = (
        args.results_dir
        or f"results-first-{args.neigh_order}"
    )

#  Best improvement GVNS
elif args.mode == "best":
    first_improvement = False
    params["neigh_order"] = args.neigh_order

    RESULTS_FOLDER = (
        args.results_dir
        or f"results-{args.neigh_order}"
    )

#  RL based GVNS
elif args.mode == "rl":
    # neigh_order is ignored in RL
    RESULTS_FOLDER = args.results_dir or "results-rl"

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

# ============================================================
# Run benchmark
# ============================================================
os.makedirs(RESULTS_FOLDER, exist_ok=True) # this is just a utility to ensure results folder exists

# run benchmark
if __name__ == "__main__":
    benchmark.benchmark_main(
        INST_FOLDER=INST_FOLDER,
        RESULTS_FOLDER=RESULTS_FOLDER,
        PARAMS=params,
        agent=agent,
        first_improvement=first_improvement,
        N_RUNS=N_RUNS
    )