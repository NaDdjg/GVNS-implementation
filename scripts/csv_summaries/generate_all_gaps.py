import csv

# Read the Best CSV file
best_data = []
with open('cost_comparison_best.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['instance']:  # Skip empty rows
            best_data.append(row)

# Read the First CSV file
first_data = []
with open('cost_comparison_first.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['instance']:  # Skip empty rows
            first_data.append(row)

# Create a mapping for first data by instance
first_dict = {row['instance']: row for row in first_data}

# Create output data with gaps from both Best and First
output_data = []
for row in best_data:
    instance = row['instance']
    output_row = {'instance': instance}
    
    methods_best = ['default', 'suggested', 'reverse', 'random', 'qlearning']
    methods_first = ['default', 'suggested', 'reverse']
    
    all_gaps = {}
    
    paper_value = round(float(row['f_paper']), 2)
    
    # Calculate gaps for Best methods
    for method in methods_best:
        value = round(float(row[f'f_best_{method}']), 2)
        gap = ((value - paper_value) / paper_value) * 100
        gap_col = f'gap_best_{method}'
        output_row[gap_col] = round(gap, 2)
        all_gaps[f'best_{method}'] = gap
    
    # Calculate gaps for First methods
    if instance in first_dict:
        first_row = first_dict[instance]
        for method in methods_first:
            value = round(float(first_row[f'f_first_{method}']), 2)
            gap = ((value - paper_value) / paper_value) * 100
            gap_col = f'gap_first_{method}'
            output_row[gap_col] = round(gap, 2)
            all_gaps[f'first_{method}'] = gap
    
    # Find best method overall (smallest absolute gap across all methods)
    best_method_key = min(all_gaps.items(), key=lambda x: abs(x[1]))[0]
    output_row['best_method'] = best_method_key
    
    output_data.append(output_row)

# Write output file
fieldnames = ['instance', 
              'gap_best_default', 'gap_best_suggested', 'gap_best_reverse', 'gap_best_random', 'gap_best_qlearning',
              'gap_first_default', 'gap_first_suggested', 'gap_first_reverse',
              'best_method']
with open('gaps_all_methods.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_data)

print("File created successfully: gaps_all_methods.csv")
print("\nFirst few rows:")
for i, row in enumerate(output_data[:10]):
    print(row)
