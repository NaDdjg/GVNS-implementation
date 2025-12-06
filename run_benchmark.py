import os
from gvns import benchmark

# parameters
N_RUNS = 1  
INST_FOLDER = "instances"      
RESULTS_FOLDER = "results"

# GVNS hyperparameters 
params = {
    "Contmax": 100,
    "p_init": 1,
    "p_max": 10,
    "Dp": 0.3,
    "w1": 0.8,    # kan 0.5, mais to have less time in the start to solve penalties
    "w2": 0.1,
    "alpha": 0.3, # kan 0.1 mais to make penalties solved in a stronger way
    "delta": 0.1, # kan 0.4 mais to keep accpetance tae worse f meshi kbira 
    "M": 1e6,
    "seed": 42,
    "Tmax": 13000  
}

# this is just a utility to ensure results folder exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# run benchmark
if __name__ == "__main__":
    benchmark.benchmark_main(
        INST_FOLDER=INST_FOLDER,
        RESULTS_FOLDER=RESULTS_FOLDER,
        PARAMS=params,
        N_RUNS=N_RUNS
    )
