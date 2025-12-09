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
    "alpha": 0.3, # kan 0.1 mais to make penalties solved in a stronger way
    "delta": 0.4, # kan 0.4 mais to keep accpetance tae worse f meshi kbira 
    "M": 1e6,
    "seed": 42,
    "Tmax": 200  
}
 