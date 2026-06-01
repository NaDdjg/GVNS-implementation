"""
Script to generate a scalability analysis plot showing average gaps for different methods
across instance sizes (50, 100, 200, 400).

Specify the CSV file path and output folder below before running.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify these paths before running
# ============================================================================
CSV_FILE_PATH = r"result_summaries\summaries\new_synch\gaps_all_methods.csv"
OUTPUT_FOLDER = r"result_summaries\plots\scalability_analysis"
# ============================================================================

# Define method names and their corresponding columns
METHODS = {
    "Q-learning": "gap_Qlearning",
    "DQN": "gap_DQN",
    "BF Selector": "gap_BF_Slector",
    "Q-learning + BF": "gap_Qlearning_BF",
    "DQN + BF": "gap_DQN_BF",
    "PPO": "gap_PPO",
}

# Define sizes in order
SIZES = [50, 100, 200, 400]

# Color palette - clean and distinct
COLORS = {
    "Q-learning": "#FF6B6B",        # Red
    "DQN": "#4ECDC4",               # Teal
    "BF Selector": "#95A5A6",       # Gray
    "Q-learning + BF": "#F39C12",   # Orange
    "DQN + BF": "#3498DB",          # Blue
    "PPO": "#2ECC71",               # Green
}

def read_and_parse_csv(filepath):
    """
    Read CSV and split into groups by instance size.
    Groups are separated by blank rows in the original file.
    """
    df = pd.read_csv(filepath)
    
    # Remove rows where all values are NaN (blank rows)
    df_clean = df.dropna(how='all')
    
    return df_clean

def split_by_size_groups(df):
    """
    Split dataframe into groups by size based on row position.
    Data is organized as 18 instances per size: [50, 100, 200, 400]
    """
    groups = {50: [], 100: [], 200: [], 400: []}
    size_order = [50, 100, 200, 400]
    rows_per_group = 18
    
    for size_idx, size in enumerate(size_order):
        start_row = size_idx * rows_per_group
        end_row = start_row + rows_per_group
        
        for i in range(start_row, min(end_row, len(df))):
            groups[size].append(df.iloc[i])
    
    return groups

def calculate_averages(groups):
    """
    Calculate average gap for each method at each size.
    Returns a dictionary: {method: {size: average_gap}}
    """
    averages = {}
    
    for method_name, column_name in METHODS.items():
        averages[method_name] = {}
        
        for size in SIZES:
            if groups[size]:
                # Convert to numeric, handling any non-numeric values
                values = []
                for row in groups[size]:
                    try:
                        val = float(row[column_name])
                        values.append(val)
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric values
                
                # Calculate mean
                if len(values) > 0:
                    averages[method_name][size] = np.mean(values)
                else:
                    averages[method_name][size] = np.nan
    
    return averages

def create_plot(averages, output_folder):
    """
    Create and save the scalability plot.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each method
    for method_name in METHODS.keys():
        sizes = list(SIZES)
        gaps = [averages[method_name].get(size, np.nan) for size in sizes]
        
        ax.plot(
            sizes, 
            gaps, 
            marker='o', 
            linewidth=2.5, 
            markersize=8,
            label=method_name,
            color=COLORS[method_name],
            alpha=0.85
        )
    
    # Add horizontal line at y=0 (GVNS baseline)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='GVNS+ (baseline)')
    
    # Formatting
    ax.set_xlabel('Instance Size (number of customers)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Relative Gap (%)', fontsize=12, fontweight='bold')
    ax.set_title('Scalability Analysis: Average Gap Relative to GVNS', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis ticks
    ax.set_xticks(SIZES)
    
    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = Path(output_folder) / "scalability_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    
    plt.close()

def main():
    """Main execution."""
    # Verify paths
    csv_path = Path(CSV_FILE_PATH)
    output_path = Path(OUTPUT_FOLDER)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output folder not found: {output_path}")
    
    print(f"Reading CSV from: {csv_path}")
    
    # Read and process data
    df = read_and_parse_csv(csv_path)
    groups = split_by_size_groups(df)
    
    # Verify we have data for all sizes
    print("\nData points per size:")
    for size in SIZES:
        count = len(groups[size])
        print(f"  Size {size}: {count} instances")
    
    # Calculate averages
    averages = calculate_averages(groups)
    
    print("\nAverage gaps by method and size:")
    for method_name in METHODS.keys():
        print(f"\n  {method_name}:")
        for size in SIZES:
            gap = averages[method_name].get(size, np.nan)
            if not np.isnan(gap):
                print(f"    Size {size}: {gap:+.2f}%")
    
    # Create and save plot
    print(f"\nGenerating plot...")
    create_plot(averages, output_path)
    
    print("\n✓ Done!")

if __name__ == "__main__":
    main()
