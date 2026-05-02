from gvns.data_generation.rl.collect_runner import collect_dataset
import time

TRAINING_BASE = "instances\\PPO_instances\\training"
VALIDATION_BASE = "instances\\PPO_instances\\validation"

# =========================
# TRAIN COLLECTION
# =========================
for size in ["50"]:
    print(f"\n===== TRAINING SIZE {size} =====")

    collect_dataset(
        instances_folder=f"{TRAINING_BASE}/{size}",
        n_runs=1,
        results_dir=f"results/dataset_collection/training/{size}",
        dataset_subdir=f"training/{size}"
    )

    time.sleep(1)


# =========================
# VALIDATION COLLECTION
# =========================
for size in ["50"]:
    print(f"\n===== VALIDATION SIZE {size} =====")

    collect_dataset(
        instances_folder=f"{VALIDATION_BASE}/{size}",
        n_runs=1,
        results_dir=f"results/dataset_collection/validation/{size}",
        dataset_subdir=f"validation/{size}"
    )

    time.sleep(1)