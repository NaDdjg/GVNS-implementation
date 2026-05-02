#!/usr/bin/env python3
"""
Script to compare best_f costs from different First improvement strategies and paper values.
Outputs a CSV with instance, f_first_default, f_first_suggested, f_first_reverse, f_paper
"""

import csv
import os

def load_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row['instance']
            data[instance] = {
                'best_f': row['best_f'],
                'time': row['time'],
                'best_f_paper': row.get('best_f_paper', ''),
                'time_paper': row.get('time_paper', '')
            }
    return data

def main():
    base_dir = 'result_summaries'
    output_dir = 'result_summaries/comparison/tables'
    files = {
        'default': os.path.join(base_dir, 'merged_summary_First_default.csv'),
        'suggested': os.path.join(base_dir, 'merged_summary_First_suggested.csv'),
        'reversed': os.path.join(base_dir, 'merged_summary_First_reversed.csv')
    }

    # Load data
    data_default = load_data(files['default'])
    data_suggested = load_data(files['suggested'])
    data_reversed = load_data(files['reversed'])

    # Get all instances
    instances = set(data_default.keys()) | set(data_suggested.keys()) | set(data_reversed.keys())

    # Prepare output for costs
    cost_output = []
    for instance in sorted(instances):
        row = {
            'instance': instance,
            'f_first_default': data_default.get(instance, {}).get('best_f', ''),
            'f_first_suggested': data_suggested.get(instance, {}).get('best_f', ''),
            'f_first_reverse': data_reversed.get(instance, {}).get('best_f', ''),
            'f_paper': data_default.get(instance, {}).get('best_f_paper', '') or
                       data_suggested.get(instance, {}).get('best_f_paper', '') or
                       data_reversed.get(instance, {}).get('best_f_paper', '')
        }
        cost_output.append(row)

    # Write cost comparison CSV
    cost_file = os.path.join(output_dir, 'cost_comparison_first.csv')
    with open(cost_file, 'w', newline='') as f:
        fieldnames = ['instance', 'f_first_default', 'f_first_suggested', 'f_first_reverse', 'f_paper']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cost_output)

    print(f"Cost comparison saved to: {cost_file}")

    # Prepare output for times
    time_output = []
    for instance in sorted(instances):
        row = {
            'instance': instance,
            'time_first_default': data_default.get(instance, {}).get('time', ''),
            'time_first_suggested': data_suggested.get(instance, {}).get('time', ''),
            'time_first_reverse': data_reversed.get(instance, {}).get('time', ''),
            'time_paper': data_default.get(instance, {}).get('time_paper', '') or
                          data_suggested.get(instance, {}).get('time_paper', '') or
                          data_reversed.get(instance, {}).get('time_paper', '')
        }
        time_output.append(row)

    # Write time comparison CSV
    time_file = os.path.join(output_dir, 'time_comparison_first.csv')
    with open(time_file, 'w', newline='') as f:
        fieldnames = ['instance', 'time_first_default', 'time_first_suggested', 'time_first_reverse', 'time_paper']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(time_output)

    print(f"Time comparison saved to: {time_file}")

if __name__ == "__main__":
    main()