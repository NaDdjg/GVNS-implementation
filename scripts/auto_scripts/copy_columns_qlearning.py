import csv

SRC = 'merged_summary_best_8.csv'
DST = 'merged_summary_qlearning-Best.csv'

# Read source
source_data = {}
with open(SRC, newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        instance = row.get('instance', '').strip()
        if instance:
            source_data[instance] = {
                'best_f_paper': row.get('best_f_paper', ''),
                'time_paper': row.get('time_paper', '')
            }

# Read destination and update
rows = []
with open(DST, newline='') as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames
    for row in reader:
        instance = row.get('instance', '').strip()
        if instance in source_data:
            row['best_f_paper'] = source_data[instance]['best_f_paper']
            row['time_paper'] = source_data[instance]['time_paper']
        rows.append(row)

# Write back
with open(DST, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Copied columns to {DST}")
