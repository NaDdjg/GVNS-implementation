import os
import argparse
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from pathlib import Path
from gvns import benchmark

# rl utilities
from gvns.rl.agents.online.qlearning_agent import QTableAgent
from gvns.rl.agents.online.dqn_agent import DQNAgent
from gvns.rl.agents.offline.ppo_agent import PPOAgent

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["first", "best", "hybrid_bf", "qlearning_fi", "qlearning_bi", "qlearning_bf", "dqn_fi", "dqn_bi", "dqn_bf", "param_controller"],
    help="Execution mode"
)

parser.add_argument(
    "--neigh_order",
    type=str,
    default="default",
    choices=["default", "reverse", "random", "suggested"],
    help="Neighborhood order (non-online-RL only)"
)

parser.add_argument(
    "--tmax",
    type=int,
    default=500,
    help="Maximum execution time per instance (seconds)"
)

parser.add_argument(
    "--processes",
    type=int,
    default=None,
    help="Number of parallel processes to use"
)

parser.add_argument(
    "--instances_dir",
    type=str,
    default="instances/New_synch/50",
    help="Path to instance folder (instances/New_synch/50, instances/New_synch/100, instances/New_synch/200)"
)

parser.add_argument(
    "--results_dir",
    type=str,
    default="results/last_execution_default",
    help="optional custom results directory"
)

args = parser.parse_args()

# ============================================================
# GVNS hyperparameters 
# neigh_order can be: "default", "reverse", "random", "suggested"
# ============================================================
N_RUNS = 2
NEIGHBORHOOD_NAMES = ['N1','N2','N3','N4','N5','N6','N7','N8']

params = {
    # logging
    "log_path": "result_summaries/debug/gvns_trace.jsonl",

    # fixed params
    "Contmax": 100,
    "p_init": 1,
    "p_max": 12,
    "Dp": 0.3,
    "w1": 0.5,    
    "w2": 0.1,
    "alpha": 0.1, 
    "delta": 0.4, 
    "M": 1e6,
    "seed": 42,

    # changeable params
    "Tmax": 1800,
    "neigh_order": "default",
    
    # ML specific params
    "use_ml": False,
    "verbose_ml": False,
    "log_ls_data": False,

    # RL / controller flags 
    "reward_mode": "qlearning",
    "use_controller": False,
    "collect_data": False
}

# ============================================================
# Mode-dependent configuration
# ============================================================
agent = None
ppo_agent = None
first_improvement = False

INST_FOLDER = Path(args.instances_dir)
if not INST_FOLDER.exists():
    raise FileNotFoundError(f"Instances directory not found: {INST_FOLDER}")

# ===============================
# Default GVNS 
# ===============================
params["Tmax"] = args.tmax
n_processes = args.processes

#  First improvement GVNS
if args.mode == "first":
    first_improvement = True
    params["neigh_order"] = args.neigh_order

    RESULTS_FOLDER = (
        args.results_dir
        or f"results/last_execution_first_{args.neigh_order}"
    )

#  Best improvement GVNS
elif args.mode == "best":
    first_improvement = False
    params["neigh_order"] = args.neigh_order

    RESULTS_FOLDER = (
        args.results_dir
        or f"results/last_execution_best_{args.neigh_order}"
    )

# =================================
# ML-based decision GVNS (hubrid FI/BI)
# =================================
elif args.mode == "hybrid_bf":
    params["use_ml"] = True
    params["neigh_order"] = args.neigh_order
    
    RESULTS_FOLDER = (
        args.results_dir
        or f"results/last_execution_ml_decision_{args.neigh_order}"
    )

# =================================
# Q-learning based GVNS (neighborhood order)
# =================================
#  Best improvement
elif args.mode == "qlearning_bi":
    # neigh_order is ignored in RL
    RESULTS_FOLDER = args.results_dir or "results/last_execution_rl"

    n_actions = len(NEIGHBORHOOD_NAMES)
    agent = QTableAgent(
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.95,
        eps_start=0.1,
        eps_min=0.01,
        eps_decay=5e-4,
        use_ucb=True,
        ucb_c=1.0,
        seed=params["seed"]
    )

#  First improvement
elif args.mode == "qlearning_fi":
    # neigh_order is ignored in RL
    RESULTS_FOLDER = args.results_dir or "results/last_execution_rl"
    first_improvement = True

    n_actions = len(NEIGHBORHOOD_NAMES)
    agent = QTableAgent(
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.95,
        eps_start=0.1,
        eps_min=0.01,
        eps_decay=5e-4,
        use_ucb=True,
        ucb_c=1.0,
        seed=params["seed"]
    )

# =================================
# DQN based GVNS (neighborhood order)
# =================================
#  Best improvement
elif args.mode == "dqn_bi":
    # neigh_order is ignored in RL
    params["reward_mode"] = "dqn"
    RESULTS_FOLDER = args.results_dir or "results/last_execution_rl"

    n_actions = len(NEIGHBORHOOD_NAMES)
    agent = DQNAgent(
        n_actions=8,
        lr=3e-4,
        gamma=0.99,
        batch_size=32,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_freq=100,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=500,
        seed =params["seed"]
    )

#  First improvement
elif args.mode == "dqn_fi":
    # neigh_order is ignored in RL
    params["reward_mode"] = "dqn"
    RESULTS_FOLDER = args.results_dir or "results/last_execution_rl"
    first_improvement = True

    n_actions = len(NEIGHBORHOOD_NAMES)
    agent = DQNAgent(
        n_actions=8,
        lr=3e-4,
        gamma=0.99,
        batch_size=32,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_freq=100,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=500,
        seed =params["seed"]
    )

# =================================
# Hybrid methods
# =================================
elif args.mode == "dqn_bf":
    # neigh_order is ignored in RL
    params["use_ml"] = True
    params["reward_mode"] = "dqn"
    RESULTS_FOLDER = args.results_dir or "results/last_execution_rl"

    n_actions = len(NEIGHBORHOOD_NAMES)
    agent = DQNAgent(
        n_actions=8,
        lr=3e-4,
        gamma=0.99,
        batch_size=32,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_freq=100,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=500,
        seed =params["seed"]
    )

elif args.mode == "qlearning_bf":
    # neigh_order is ignored in RL
    params["use_ml"] = True    
    RESULTS_FOLDER = args.results_dir or "results/last_execution_rl"

    n_actions = len(NEIGHBORHOOD_NAMES)
    agent = QTableAgent(
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.95,
        eps_start=0.1,
        eps_min=0.01,
        eps_decay=5e-4,
        use_ucb=True,
        ucb_c=1.0,
        seed=params["seed"]
    )

elif args.mode == "param_controller":
    params["use_controller"] = True
    RESULTS_FOLDER = args.results_dir or "results/last_execution_param_controller"

    # load dataset metadata
    data = np.load("gvns/datasets/param_controller_cleaned/train_clean.npz")
    state_dim = data["states"].shape[1]
    n_actions = 8  
    feature_names = data["feature_names"].tolist()

    ppo_agent = PPOAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        feature_names=feature_names,
    )

    # load trained weights 
    ppo_agent.load("gvns/rl/notebooks/models/ppo_param_controller.pt")


# ============================================================
# Run benchmark
# ============================================================
os.makedirs(RESULTS_FOLDER, exist_ok=True) # this is just a utility to ensure results folder exists

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
        N_RUNS=N_RUNS,
        agent=agent,
        ppo_agent=ppo_agent,
        first_improvement=first_improvement,
        n_processes=n_processes
    )