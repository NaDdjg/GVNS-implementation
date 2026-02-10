import pandas as pd
import numpy as np
from pathlib import Path

def verify_dataset(csv_path):
    """Verify the dataset by checking columns info, nulls, types, and value ranges."""
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print(f"Dataset Verification: {csv_path}")
    print("=" * 80)
    
    # Basic info
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names and Info:")
    print("-" * 80)
    
    for col in df.columns:
        print(f"\nColumn: '{col}'")
        print(f"   Data Type: {df[col].dtype}")
        print(f"   Non-null Count: {df[col].notna().sum()}")
        print(f"   Null Count: {df[col].isna().sum()}")
        print(f"   Null Percentage: {(df[col].isna().sum() / len(df)) * 100:.2f}%")
        
        # Get stats based on data type
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            print(f"   Min Value: {df[col].min()}")
            print(f"   Max Value: {df[col].max()}")
            print(f"   Mean: {df[col].mean():.4f}")
            print(f"   Median: {df[col].median():.4f}")
            print(f"   Std Dev: {df[col].std():.4f}")
        else:
            unique_count = df[col].nunique()
            print(f"   Unique Values: {unique_count}")
            if unique_count <= 10:
                print(f"   Values: {df[col].unique().tolist()}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    print(df.describe().T)
    
    # Check for duplicates
    print("\n" + "=" * 80)
    print("Duplicate Check:")
    print("=" * 80)
    duplicate_rows = df.duplicated().sum()
    print(f"Total Duplicate Rows: {duplicate_rows}")
    
    # Check for any missing values summary
    print("\n" + "=" * 80)
    print("Missing Values Summary:")
    print("=" * 80)
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print(missing_summary[missing_summary > 0])
    else:
        print("No missing values found!")
    
    # Data type summary
    print("\n" + "=" * 80)
    print("Data Types Summary:")
    print("=" * 80)
    print(df.dtypes)
    
    return df


if __name__ == "__main__":
    # Path to the learning dataset
    csv_path = Path(__file__).parent.parent.parent / "gvns" / "ls_data" / "learning_dataset.csv"
    
    if csv_path.exists():
        df = verify_dataset(csv_path)
    else:
        print(f"Error: Dataset not found at {csv_path}")