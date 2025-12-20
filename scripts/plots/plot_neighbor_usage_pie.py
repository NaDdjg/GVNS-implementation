#!/usr/bin/env python3
"""
Script to create pie charts for neighbor usage from neigh_usage_global.csv files.
"""

import csv
import os
import matplotlib.pyplot as plt
import glob

def plot_pie_chart(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nid = int(row['Neighborhood_ID'])
                usage = float(row['Usage_Count'])
                pct = float(row['percentage'])
                data.append((nid, usage, pct))
            except:
                continue

    if not data:
        print(f"No data in {file_path}")
        return

    # Sort by usage descending
    data.sort(key=lambda x: x[1], reverse=True)

    labels = [f'ID {nid}' for nid, _, _ in data]
    sizes = [pct * 100 for _, _, pct in data]  # percentage as fraction, but for pie, use as is? Wait, percentage is already 0-1? Wait.

    # percentage is like 0.624, so sizes = [pct for _, _, pct in data], but for pie, it's proportions.

    # But to show percentage, sizes = [pct * 100 for ...] no, pie takes fractions.

    # Actually, plt.pie(sizes, autopct='%1.1f%%') where sizes are the values.

    # Since percentage is the proportion, sizes = [usage for _, usage, _ in data]

    # But to show the percentage, use autopct.

    sizes = [usage for _, usage, _ in data]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    title = os.path.basename(file_path).replace('_global.csv', '').replace('_', ' ').title()
    ax.set_title(f'Neighbor Usage: {title}')

    output_file = file_path.replace('.csv', '.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved pie chart to: {output_file}")

def main():
    base_dir = 'result_summaries/neigh_usage'
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.")
        return

    # Find all *_global.csv files recursively
    pattern = os.path.join(base_dir, '**', '*_global.csv')
    files = glob.glob(pattern, recursive=True)

    if not files:
        print("No *_global.csv files found.")
        return

    for file_path in files:
        plot_pie_chart(file_path)

if __name__ == "__main__":
    main()