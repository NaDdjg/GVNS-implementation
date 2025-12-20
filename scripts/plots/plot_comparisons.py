"""
Script to create bar charts comparing costs and times from First improvement strategies.
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np

def load_comparison_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row['instance']
            data[instance] = row
    return data

def plot_comparison(data, value_key, title, ylabel, output_file):
    instances = list(data.keys())
    methods = ['f_best_default', 'f_best_suggested', 'f_best_reverse', 'f_paper'] if 'f' in value_key else ['time_best_default', 'time_best_suggested', 'time_best_reverse', 'time_paper']
    
    # Prepare data
    values = {method: [] for method in methods}
    for instance in instances:
        for method in methods:
            val = data[instance].get(method, '')
            try:
                values[method].append(float(val))
            except:
                values[method].append(np.nan)

    # Plot
    x = np.arange(len(instances))
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(10, len(instances) * 0.8), 10))  # Wide and tall enough

    colors = ["#800000", "#4400FF", "#92B205", '#FF00FF']
    for i, method in enumerate(methods):
        ax.bar(x + i * width, values[method], width, label=method.replace('_', ' ').title(), color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Instance')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(instances, rotation=45, ha='right')

    # Set y-axis to show small differences
    all_values = []
    for method in methods:
        for v in values[method]:
            if np.isfinite(v):
                all_values.append(v)
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        margin = (max_val - min_val) * 0.05
        ax.set_ylim(max(0, min_val - margin), max_val + margin)

    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_file}")

def main():
    base_dir = 'result_summaries/comparison/tables'
    plots_dir = 'result_summaries/comparison/plots'

    # Load data
    cost_data = load_comparison_data(os.path.join(base_dir, 'cost_comparison_best.csv'))
    time_data = load_comparison_data(os.path.join(base_dir, 'time_comparison_best.csv'))

    # Plot cost
    plot_comparison(
        cost_data,
        'f',
        'Cost Comparison: Best Improvement Strategies',
        'Cost (best_f)',
        os.path.join(plots_dir, 'cost_comparison_best.png')
    )

    # Plot time
    plot_comparison(
        time_data,
        'time',
        'Time Comparison: Best Improvement Strategies',
        'Time (seconds)',
        os.path.join(plots_dir, 'time_comparison_best.png')
    )

if __name__ == "__main__":
    main()