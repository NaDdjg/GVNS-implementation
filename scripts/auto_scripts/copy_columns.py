import csv

# Read merged_summary.csv to extract best_f_paper and time_paper
source_data = {}
with open('merged_summary.csv', newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        instance = row.get('instance', '').strip()
        if instance:
            source_data[instance] = {
                'best_f_paper': row.get('best_f_paper', ''),
                'time_paper': row.get('time_paper', '')
            }

# Read merged_summary_new.csv and update columns
rows = []
fieldnames = None
with open('result_summaries/merged_summary_qlearning_best.csv', newline='') as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames
    for row in reader:
        instance = row.get('instance', '').strip()
        if instance in source_data:
            row['best_f_paper'] = source_data[instance]['best_f_paper']
            row['time_paper'] = source_data[instance]['time_paper']
        rows.append(row)

# Write back to merged_summary_new.csv
with open('result_summaries/merged_summary_qlearning_best.csv', 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
