import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 3:
    print("Usage: python plot_neigh_usage.py <input_folder> <output_folder>")
    sys.exit(1)

input_folder = sys.argv[1]
output_folder = sys.argv[2]

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if '_neigh_improvements_rl.csv' in file:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        
        # Find column names case-insensitively
        iteration_col = None
        neigh_id_col = None
        for col in df.columns:
            if col.lower() == 'iteration':
                iteration_col = col
            elif col.lower() == 'neighborhood_id':
                neigh_id_col = col
        
        if iteration_col is None or neigh_id_col is None:
            print(f"Required columns not found in {file}. Columns: {df.columns.tolist()}")
            continue
        
        plt.figure(figsize=(12, 6))
        plt.bar(df[iteration_col], 1, bottom=df[neigh_id_col], width=1, align='edge', color='skyblue', edgecolor='black')
        plt.xlabel('Iteration')
        plt.ylabel('Neighborhood')
        plt.title('Neighborhood Usage: ' + file.replace('_neigh_improvements_rl.csv', ''))
        plt.ylim(0, 8)
        plt.yticks([i + 0.5 for i in range(8)], ['N' + str(i+1) for i in range(8)])
        plt.grid(True, axis='x')
        plt.tight_layout()
        try:
            plt.savefig(output_folder, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_folder}")
        except Exception as e:
            print(f"Failed to save plot to {output_folder}: {e}")
            
        plt.close()