import sys
import subprocess
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

INSTANCES_DIR = "instances/demo_instances"

TMAX = 60

MODELS = [
    (
        "GVNS_PLUS",
        {
            "mode": "first",
            "extra": ["--neigh_order", "default"]
        }
    ),

    (
        "PPO",
        {
            "mode": "param_controller",
            "extra": []
        }
    ),

    (
        "QLEARNING",
        {
            "mode": "qlearning_fi",
            "extra": []
        }
    ),

    (
        "DQN",
        {
            "mode": "dqn_fi",
            "extra": []
        }
    ),

    (
        "BF_SELECTOR",
        {
            "mode": "hybrid_bf",
            "extra": ["--neigh_order", "default"]
        }
    ),

    (
        "QLEARNING_BF",
        {
            "mode": "qlearning_bf",
            "extra": []
        }
    ),

    (
        "DQN_BF",
        {
            "mode": "dqn_bf",
            "extra": []
        }
    )
]

BASE_RESULTS = Path("results/demo")

# ============================================================
# Runner
# ============================================================

def run_model(index, total, model_name, config):

    result_dir = BASE_RESULTS / model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "run_benchmark.py",

        "--mode",
        config["mode"],

        "--tmax",
        str(TMAX),

        "--instances_dir",
        INSTANCES_DIR,

        "--results_dir",
        str(result_dir),

        "--processes",
        "3"
    ]

    cmd.extend(config["extra"])

    print("\n")
    print("=" * 60)
    print(f"[{index}/{total}] Running {model_name}")
    print("=" * 60)

    print("\nCommand:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)

    print()
    print(f"✓ Finished {model_name}")
    print()


# ============================================================
# Main
# ============================================================

def main():

    BASE_RESULTS.mkdir(parents=True, exist_ok=True)

    total = len(MODELS)

    print()
    print("=" * 60)
    print("PROJECT DEMONSTRATION")
    print("=" * 60)
    print(f"Instances folder : {INSTANCES_DIR}")
    print(f"Results folder   : {BASE_RESULTS}")
    print(f"Models           : {total}")
    print()

    for idx, (model_name, config) in enumerate(MODELS, start=1):
        run_model(idx, total, model_name, config)

    print()
    print("=" * 60)
    print("ALL DEMO RUNS FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()