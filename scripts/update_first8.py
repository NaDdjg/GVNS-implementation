import csv
from decimal import Decimal, InvalidOperation

SRC = 'merged_summary_best_8.csv'
DST = 'merged_summary_First_8.csv'

def parse_decimal(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

# Step 1: Read source columns
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

# Step 2: Read destination, update columns, compute gaps
rows = []
with open(DST, newline='') as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames
    for row in reader:
        instance = row.get('instance', '').strip()
        if instance in source_data:
            row['best_f_paper'] = source_data[instance]['best_f_paper']
            row['time_paper'] = source_data[instance]['time_paper']
        
        # Compute gap
        best_f = parse_decimal(row.get('best_f', ''))
        best_f_paper = parse_decimal(row.get('best_f_paper', ''))
        gap = ''
        if best_f is not None and best_f_paper and best_f_paper != 0:
            gap = (best_f - best_f_paper) / best_f_paper * 100
            gap = f"{gap.quantize(Decimal('0.01'))}"
        row['gap_f_with_paper'] = gap
        
        rows.append(row)

# Step 3: Write back
with open(DST, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {DST}: copied columns and computed gaps.")
