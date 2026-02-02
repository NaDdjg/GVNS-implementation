import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,         # ranking quality
    brier_score_loss,      # probability calibration
    classification_report,  # detailed classification metrics
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

required_cols = {"success", "strategy", "instance_name"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -----------------------------
# Group split (by instance)
# -----------------------------
groups = df["instance_name"]

# -----------------------------
# Features / target
# -----------------------------
DROP_COLS = [
    "delta",
    "rel_delta",
    "success",
    "instance_name",
]

X = df.drop(columns=DROP_COLS)
y = df["success"]

feature_names = X.columns.tolist()

# -----------------------------
# Train / Test split
# -----------------------------
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# -----------------------------
# Train classifier
# -----------------------------
model = RandomForestClassifier(**RF_PARAMS)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
p_train = model.predict_proba(X_train)[:, 1]
p_test = model.predict_proba(X_test)[:, 1]

metrics = {
    "train": {
        "roc_auc": roc_auc_score(y_train, p_train),
        "brier": brier_score_loss(y_train, p_train),
    },
    "test": {
        "roc_auc": roc_auc_score(y_test, p_test),
        "brier": brier_score_loss(y_test, p_test),
    },
    "n_train": len(train_idx),
    "n_test": len(test_idx),
    "pos_rate_train": y_train.mean(),
    "pos_rate_test": y_test.mean(),
}

# -----------------------------
# Save artifacts
# -----------------------------
joblib.dump(model, OUTPUT_DIR / "rf_success_classifier.pkl")

with open(OUTPUT_DIR / "rf_classifier_features.json", "w") as f:
    json.dump(feature_names, f, indent=2)

with open(OUTPUT_DIR / "rf_classifier_report.txt", "w") as f:
    f.write("RandomForestClassifier – Success Prediction\n")
    f.write("=" * 50 + "\n\n")

    for split in ["train", "test"]:
        f.write(f"{split.upper()} METRICS\n")
        for k, v in metrics[split].items():
            f.write(f"{k}: {v:.6f}\n")
        f.write("\n")

    f.write(f"Train samples: {metrics['n_train']}\n")
    f.write(f"Test samples: {metrics['n_test']}\n")
    f.write(f"Positive rate (train): {metrics['pos_rate_train']:.4f}\n")
    f.write(f"Positive rate (test): {metrics['pos_rate_test']:.4f}\n\n")

    f.write("RF Parameters:\n")
    for k, v in RF_PARAMS.items():
        f.write(f"{k}: {v}\n")

# -----------------------------
# Console summary
# -----------------------------
print("Classifier training completed.")
print(f"Train ROC-AUC: {metrics['train']['roc_auc']:.4f}")
print(f"Test  ROC-AUC: {metrics['test']['roc_auc']:.4f}")
print(f"Model saved to: {OUTPUT_DIR}")
