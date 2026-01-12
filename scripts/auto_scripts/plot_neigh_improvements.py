import os
import pandas as pd
import matplotlib.pyplot as plt

# Configurable input folder
input_folder = r"results\with RL\qlearning\First_alpha01_gamma095\neigh_usage"

# Output folder for plots
output_folder = r"result_summaries\neighborhoods_usage_qlearning_plots"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Find all files matching the pattern
pattern = "*_neigh_improvements*.csv"
files = [f for f in os.listdir(input_folder) if f.endswith("_neigh_improvements.csv") or "_neigh_improvements" in f]

for file in files:
    file_path = os.path.join(input_folder, file)
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Assuming columns: iteration, neighborhood_id, neighborhood_name, F_before, F_after
    # We want to plot neighborhood_id or name per iteration
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['neighborhood_id'], marker='o', linestyle='-')
    plt.title(f'Neighborhood Usage Over Iterations - {file}')
    plt.xlabel('Iteration')
    plt.ylabel('Neighborhood ID')
    plt.xticks(df['iteration'])
    plt.grid(True)
    
    # Save the plot
    plot_filename = file.replace('.csv', '_plot.png')
    plot_path = os.path.join(output_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved for {file} at {plot_path}")