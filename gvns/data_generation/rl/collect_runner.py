import os
import time

from gvns.data_generation.rl.collector import GVNSCollector
from gvns.data_generation.rl.params.controller import ParamController
from gvns.benchmark import run_folder

# =========================================================
# Time limits per instance size 
# =========================================================
TIME_LIMITS = {
    50: 900,     # 15 min
    100: 1800,   # 30 min
    200: 2700,   # 45 min
    400: 3600,   # 60 min
}

FEATURE_NAMES = [
    "gvns_iter",
    "stagnation_ls",
    "time_elapsed_ratio",
    "time_left_ratio",
    "improv_rate_W",
    "avg_delta_W",
    "max_delta_W",
    "avg_delta_W_norm",
    "max_delta_W_norm",
    "ls_fail_ratio_W",
    "PV",
    "pen_ratio",
    "penT_ratio",
    "penS_ratio",
    "k",
    "k_norm",
    "stagnation_time_pressure",

    # instance
    "n_customers",
    "n_vehicles",
    "horizon",
    "sync_ratio",
    "avg_tw_width",
    "min_tw_width",
    "tw_tightness",
    "early_tw_ratio",
    "late_tw_ratio",

    # structure
    "route_len_mean",
    "route_len_std",
    "route_len_max_ratio",

    # time
    "tw_violation_ratio",
    "avg_lateness_norm",
    "waiting_time_ratio",

    # trends
    "improv_trend",
    "delta_trend",

    # controller
    "param_shaking",
    "param_acceptance",
    "param_penalty",
    "param_stagnation_limit",

    # extra
    "elapsed",
    "neigh_id"
]

def infer_time_limit(instances_folder: str) -> int:
    """
    Infer time limit based on folder name.
    """
    folder = str(instances_folder)

    for size, t in TIME_LIMITS.items():
        if str(size) in folder:
            return t

    # fallback
    return 300

def collect_dataset(
    instances_folder: str,
    n_runs: int = 3,
    results_dir: str = "results_collect",
    dataset_subdir: str = "training",
    params: dict = None,
):
    """
    Run GVNS on instances and collect RL dataset.
    """

    print("\n=== DATASET COLLECTION START ===")
    print(f"Instances folder: {instances_folder}")
    print(f"Runs per instance: {n_runs}")

    # =========================================================
    # Dataset base path
    # =========================================================
    dataset_dir = os.path.join(
        "gvns", "datasets", "param_controller", dataset_subdir
    )
    os.makedirs(dataset_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_save_path = os.path.join(dataset_dir, f"dataset_{timestamp}.npz")

    print(f"Dataset folder: {dataset_dir}")

    # =========================================================
    # Collector TEMPLATE (real ones created per process)
    # =========================================================
    collector = GVNSCollector(
        feature_names=FEATURE_NAMES,
        save_path=base_save_path
    )

    # =========================================================
    # Controller TEMPLATE
    # =========================================================
    param_controller = ParamController()

    # =========================================================
    # Params
    # =========================================================
    if params is None:
        Tmax = infer_time_limit(instances_folder)

        params = {
            # GVNS core
            "Contmax": 100,
            "p_max": 12,
            "Dp": 0.3,
            "w1": 0.5,
            "w2": 0.1,
            "alpha": 0.1,
            "delta": 0.4,
            "M": 1e6,
            "seed": 42,

            # runtime
            "Tmax": Tmax,
            "neigh_order": "default",

            # ML disabled
            "use_ml": False,
            "verbose_ml": False,
            "log_ls_data": False,

            # RL flags 
            "use_controller": True,
            "collect_data": True,
        }

    print(f"Time limit per run: {params['Tmax']} seconds")

    # =========================================================
    # Run parallel benchmark
    # =========================================================
    run_folder(
        folder=instances_folder,
        n_runs=n_runs,
        params=params,
        agent=None,
        first_improvement=True,
        results_dir=results_dir,
        qtables_dir=None,
        collector=collector,
        param_controller=param_controller
    )

    print("\n=== DATASET COLLECTION DONE ===")
    print(f"Datasets saved in: {dataset_dir}")