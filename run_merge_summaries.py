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
        # ("results/without_RL/predefined_order/Best", "result_summaries/summaries/liu_instances/merged_summary_Best_default.csv"),
        # ("results/without_RL/predefined_order/First", "result_summaries/summaries/liu_instances/merged_summary_First_default.csv"),
        # ("results/without_RL/suggested_order/Best", "result_summaries/summaries/liu_instances/merged_summary_Best_suggested.csv"),
        # ("results/without_RL/suggested_order/First", "result_summaries/summaries/liu_instances/merged_summary_First_suggested.csv"),
        # ("results/without_RL/reverse_order/First", "result_summaries/summaries/liu_instances/merged_summary_First_reversed.csv"),
        # ("results/without_RL/reverse_order/Best", "result_summaries/summaries/liu_instances/merged_summary_Best_reversed.csv"),
        # ("results/without_RL/random_order/Best", "result_summaries/summaries/liu_instances/merged_summary_Best_random.csv"),
        # ("results/without_RL/random_order/First", "result_summaries/summaries/liu_instances/merged_summary_First_random.csv"),
        # ("results/with_RL/qlearning/Best", "result_summaries/summaries/liu_instances/merged_summary_qlearning_Best.csv"),
        # ("results/without_RL/predefined_order/ML_decision", "result_summaries/summaries/liu_instances/merged_summary_ML_default.csv"),

        # ("results/without_RL/predefined_order/First/Bredstrom", "result_summaries/summaries/bredstrom_instances/merged_summary_First_default.csv"),
        # ("results/without_RL/predefined_order/Best/Bredstrom", "result_summaries/summaries/bredstrom_instances/merged_summary_Best_default.csv"),

        # ("results/without_RL/predefined_order/First/400", "result_summaries/summaries/new_synch/merged_summary_GVNS_new.csv"),
        ("results/with_RL/qlearning/First/400", "result_summaries/summaries/new_synch/merged_summary_First_qlearning.csv"),
        # ("results/without_RL/predefined_order/ML_decision/400", "result_summaries/summaries/new_synch/merged_summary_ML_default.csv"),
        # ("results/with_RL/dqn/First/200", "result_summaries/summaries/new_synch/merged_summary_First_dqn.csv"),
    ]

    for input_dir, output_file in merges:
        print(f"Running merge for {input_dir} -> {output_file}")
        merge_summaries(input_dir, output_file)
        print(f"Completed: {output_file}\n")

if __name__ == "__main__":
    main()