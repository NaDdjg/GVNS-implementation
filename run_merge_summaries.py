"""
Script to run all merge_summaries operations for different result directories.
"""

import os
import sys

# Add the current directory to sys.path to import merge_summaries
sys.path.insert(0, os.path.dirname(__file__))

from scripts.csv_summaries.merge_summaries import merge_summaries

def main():
    # Define the merge operations
    merges = [
        # ("results/methods/Hybrid/DQN_BF/50", "result_summaries/summaries/new_synch/summary_DQN_BF.csv"),
        # ("results/methods/Hybrid/Qlearning_BF/50", "result_summaries/summaries/new_synch/summary_Qlearning_BF.csv"),   
        ("results/methods/Hybrid/Qlearning_BF/400", "result_summaries/summaries/new_synch/summary_Qlearning_BF_400.csv"),
        # ("results/methods/Hybrid/Qlearning_BF/100", "result_summaries/summaries/new_synch/summary_Qlearning_BF_100.csv"),   
        # ("results/methods/Hybrid/DQN_BF/200", "result_summaries/summaries/new_synch/summary_DQN_BF_200.csv"),
        # ("results/methods/Hybrid/Qlearning_BF/200", "result_summaries/summaries/new_synch/summary_Qlearning_BF_200.csv"),   

    ]

    for input_dir, output_file in merges:
        print(f"Running merge for {input_dir} -> {output_file}")
        merge_summaries(input_dir, output_file)
        print(f"Completed: {output_file}\n")

if __name__ == "__main__":
    main()