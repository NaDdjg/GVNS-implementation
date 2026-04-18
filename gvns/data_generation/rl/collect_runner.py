import os

from gvns.data_generation.rl.collector import GVNSCollector
from gvns.data_generation.rl.params.controller import ParamController
from gvns.benchmark import run_folder_parallel

def collect_dataset(
    instances_folder: str,
    save_path: str,
    n_runs: int = 3,
    action_dim: int = 8,
    results_dir: str = "results_collect",
    n_processes: int = None,
    params: dict = None,
):
    """
    Run GVNS on a folder of instances and collect RL transitions.

    Parameters
    ----------
    instances_folder : str
        Path to folder containing instances.
    save_path : str
        Where to save the dataset (.npz).
    n_runs : int
        Number of runs per instance.
    action_dim : int
        Number of discrete actions.
    results_dir : str
        Where to store GVNS traces (optional).
    n_processes : int
        Number of parallel workers.
    params : dict
        GVNS parameters.
    """

    print("\n=== DATASET COLLECTION START ===")
    print(f"Instances folder: {instances_folder}")
    print(f"Save path: {save_path}")
    print(f"Runs per instance: {n_runs}")
    print(f"Processes: {n_processes}")

    # =========================================================
    # 1. Collector
    # =========================================================
    collector = GVNSCollector(
        action_dim=action_dim,
        save_path=save_path
    )

    # =========================================================
    # 2. Parameter controller
    # =========================================================
    param_controller = ParamController()

    # =========================================================
    # 3. Default params (safe baseline)
    # =========================================================
    if params is None:
        params = {
            "seed": 42,
            "time_limit": 60,   # will change this depending on the instance size
            "k_max": 10,
        }

    # =========================================================
    # 4. Run GVNS benchmark (parallel)
    # =========================================================
    run_folder_parallel(
        folder=instances_folder,
        n_runs=n_runs,
        params=params,
        agent=None,  # no RL agent here
        first_improvement=True,
        results_dir=results_dir,
        qtables_dir=None,
        n_processes=n_processes,

        collector=collector,
        param_controller=param_controller
    )

    print("\n=== DATASET COLLECTION DONE ===")
    print(f"Dataset saved at: {save_path}")


# how to use:
# from gvns.rl.data.collect_runner import collect_dataset

# collect_dataset(
#     instances_folder="instances/50_100",
#     save_path="datasets/rl_dataset.npz",
#     n_runs=3,
#     n_processes=8
# )