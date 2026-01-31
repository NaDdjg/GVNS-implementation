import os
import pandas as pd

# =========================================================
# Paths
# =========================================================
INPUT_PATH = os.path.join(
    "gvns", "ls_data", "merged_ls_decisions.csv"
)

OUTPUT_PATH = os.path.join(
    "gvns", "ls_data", "learning_dataset.csv"
)

# =========================================================
# Load merged dataset
# =========================================================
print("Loading merged LS decision dataset...")
df = pd.read_csv(INPUT_PATH)

print(f"Total rows loaded: {len(df)}")
print("Columns:")
print(df.columns.tolist())

# =========================================================
# Basic sanity checks
# =========================================================
required_cols = {
    "strategy",   # 0 = FI, 1 = BI
    "delta",      # improvement magnitude
    "improved",   # binary flag
    "F_before",
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =========================================================
# filter, to keep only improving LS calls
# =========================================================
print("Filtering to improving LS decisions only...")
df = df[df["improved"] == 1].copy()

print(f"Remaining rows after filtering: {len(df)}")

# =========================================================
# Create relative improvement target
# =========================================================
print("\nCreating rel_delta target...")
df["rel_delta"] = df["delta"] / (df["F_before"] + 1e-6)

# df["delta"] = df["delta"].clip(lower=0.0)
# df["rel_delta"] = df["rel_delta"].clip(lower=0.0)

print("rel_delta statistics:")
print(df["rel_delta"].describe())

# =========================================================
# Drop non-learning columns
# =========================================================
DROP_COLUMNS = [
    "source_file",   # will just keep instance_name
    "mode",          # FI / BI string version (strategy is numeric)
    "improved",      # already filtered
    "neigh_name",
    "ls_call_id",

    "F_before",
    "F_after",

    "iter_gvns"      # redundant
]

existing_drop = [c for c in DROP_COLUMNS if c in df.columns]
df.drop(columns=existing_drop, inplace=True)

# =========================================================
# Final checks
# =========================================================
# 1. No NaNs
nan_counts = df.isna().sum()
if nan_counts.any():
    print("WARNING: NaNs detected:")
    print(nan_counts[nan_counts > 0])
    raise ValueError("Dataset contains NaNs. Fix before training.")

# 2. Strategy sanity
print("\nRows per strategy:")
print(df["strategy"].value_counts())

# 3. Delta sanity
print("\nDelta statistics:")
print(df["delta"].describe())

print("\nFinal feature overview:")
print(df.describe().T[["min", "mean", "max"]])

# =========================================================
# Save learning dataset
# =========================================================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nDataset ready for learning.")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Final shape: {df.shape}")