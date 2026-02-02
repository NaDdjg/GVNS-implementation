import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("gvns/ls_data/learning_dataset.csv")
OUTPUT_DIR = Path("gvns/ml/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=16,
    min_samples_leaf=30,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

# sanity checks
required_cols = {"delta", "rel_delta", "strategy"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------------
# Group split (by instance)
# -----------------------------
if "instance_name" not in df.columns:
    raise ValueError(
        "instance_name column is required for grouped split "
        "(should be kept for splitting, not as feature)"
    )

groups = df["instance_name"]

# Features: drop targets and non-feature columns
DROP_COLS = [
    "delta",
    "rel_delta",
    "instance_name",  # used only for grouping
]

X = df.drop(columns=DROP_COLS)
y = df["rel_delta"]

feature_names = X.columns.tolist()

# -----------------------------
# Train / Test split
# -----------------------------
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# -----------------------------
# Train model
# -----------------------------
model = RandomForestRegressor(**RF_PARAMS)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

metrics = {
    "train": {
        "MAE": mean_absolute_error(y_train, y_pred_train),
        "RMSE": root_mean_squared_error(y_train, y_pred_train),
        "R2": r2_score(y_train, y_pred_train),
    },
    "test": {
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "RMSE": root_mean_squared_error(y_test, y_pred_test),
        "R2": r2_score(y_test, y_pred_test),
    },
    "n_train": len(train_idx),
    "n_test": len(test_idx),
    "n_features": X.shape[1],
    "rf_params": RF_PARAMS,
}

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(model, OUTPUT_DIR / "rf_delta_regressor.pkl")

with open(OUTPUT_DIR / "rf_regressor_features.json", "w") as f:
    json.dump(feature_names, f, indent=2)

with open(OUTPUT_DIR / "rf_regressor_report.txt", "w") as f:
    f.write("RandomForestRegressor – log(delta)\n")
    f.write("=" * 40 + "\n\n")
    for split in ["train", "test"]:
        f.write(f"{split.upper()} METRICS\n")
        for k, v in metrics[split].items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\n")
    f.write(f"Train samples: {metrics['n_train']}\n")
    f.write(f"Test samples: {metrics['n_test']}\n")
    f.write(f"Features: {metrics['n_features']}\n\n")
    f.write("RF Parameters:\n")
    for k, v in RF_PARAMS.items():
        f.write(f"{k}: {v}\n")

# -----------------------------
# Console summary
# -----------------------------

print("Training completed.")
print(f"Train R²: {metrics['train']['R2']:.4f}")
print(f"Test  R²: {metrics['test']['R2']:.4f}")
print(f"Model saved to: {OUTPUT_DIR}")
