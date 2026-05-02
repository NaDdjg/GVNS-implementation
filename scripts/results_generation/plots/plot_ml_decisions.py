import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_ml_decisions():
    """
    Reads decision trace CSV files and generates line plots for each instance.
    Plots show decisions (FI or BI) over time and iterations.
    """
    
    # Paths
    input_dir = "results/without_RL/predefined_order/ML_decision/decision_traces"
    output_dir_time = "result_summaries/ml_policy_results/decison_over_time"
    output_dir_iterations = "result_summaries/ml_policy_results/decision_over_iterations"
    
    # Create output directories if they don't exist
    os.makedirs(output_dir_time, exist_ok=True)
    os.makedirs(output_dir_iterations, exist_ok=True)
    
    # Get all decision trace CSV files
    trace_files = [
        f for f in os.listdir(input_dir)
        if f.endswith("_ml_decisions_trace.csv")
    ]
    
    if not trace_files:
        print(f"[ERROR] No decision trace files found in {input_dir}")
        return
    
    for trace_file in trace_files:
        # Extract instance name (remove __ml_decisions_trace.csv)
        instance_name = trace_file.replace("__ml_decisions_trace.csv", "")
        
        file_path = os.path.join(input_dir, trace_file)
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            continue
        
        if "decision" not in df.columns:
            print(f"[WARNING] No 'decision' column in {trace_file}. Skipped.")
            continue
        
        # Create iteration numbers (row 1 = iteration 1)
        iterations = np.arange(1, len(df) + 1)
        
        # Convert decision to numeric (FI=1, BI=0)
        decisions = (df["decision"] == "FI").astype(int)
        
        # Plot 1: Decision over time_elapsed_ratio
        if "time_elapsed_ratio" in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot FI points
            fi_mask = df["decision"] == "FI"
            bi_mask = df["decision"] == "BI"
            
            ax.plot(df.loc[fi_mask, "time_elapsed_ratio"], 
                   df.loc[fi_mask].index.map(lambda x: 1),
                   'o-', color='blue', label='FI', linewidth=2, markersize=4)
            ax.plot(df.loc[bi_mask, "time_elapsed_ratio"], 
                   df.loc[bi_mask].index.map(lambda x: 0),
                   'o-', color='red', label='BI', linewidth=2, markersize=4)
            
            # Alternative: create a continuous line
            ax.plot(df["time_elapsed_ratio"], decisions, 
                   'o-', linewidth=2, markersize=4, 
                   color='green', alpha=0.3, label='Decision')
            
            # Use scatter with colors for better visualization
            colors = ['blue' if dec == 'FI' else 'red' for dec in df["decision"]]
            ax.scatter(df["time_elapsed_ratio"], decisions, c=colors, s=20, zorder=3)
            
            # Redraw line
            ax.plot(df["time_elapsed_ratio"], decisions, 'k-', alpha=0.2, linewidth=1)
            
            ax.set_xlabel("Time Elapsed Ratio", fontsize=12)
            ax.set_ylabel("Decision", fontsize=12)
            ax.set_title(f"{instance_name} - Decision over Time", fontsize=14)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["BI", "FI"])
            ax.grid(True, alpha=0.3)
            ax.legend(["FI", "BI"], loc='best')
            
            output_file = os.path.join(output_dir_time, f"{instance_name}_decision_over_time.png")
            plt.tight_layout()
            plt.savefig(output_file, dpi=100)
            plt.close()
            print(f"[OK] Saved: {output_file}")
        
        # Plot 2: Decision over iterations
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['blue' if dec == 'FI' else 'red' for dec in df["decision"]]
        ax.scatter(iterations, decisions, c=colors, s=20, zorder=3)
        
        # Draw line
        ax.plot(iterations, decisions, 'k-', alpha=0.2, linewidth=1)
        
        ax.set_xlabel("Local Search Iteration", fontsize=12)
        ax.set_ylabel("Decision", fontsize=12)
        ax.set_title(f"{instance_name} - Decision over Iterations", fontsize=14)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["BI", "FI"])
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='FI'),
                          Patch(facecolor='red', label='BI')]
        ax.legend(handles=legend_elements, loc='best')
        
        output_file = os.path.join(output_dir_iterations, f"{instance_name}_decision_over_iterations.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=100)
        plt.close()
        print(f"[OK] Saved: {output_file}")


if __name__ == "__main__":
    plot_ml_decisions()
