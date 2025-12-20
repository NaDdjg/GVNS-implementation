import csv

IN = 'merged_summary_best_8.csv'

# Data-row indices to move (1-based as requested)
MOVE_INDICES = [4,5,12,16,17,21]

with open(IN, newline='') as fh:
    reader = csv.reader(fh)
    rows = list(reader)

if not rows:
    raise SystemExit('Empty file')

header = rows[0]
data = rows[1:]

# Convert to 0-based indices for data list
move_zero = [i-1 for i in MOVE_INDICES]

moved = []
remaining = []
for idx, row in enumerate(data):
    if idx in move_zero:
        moved.append(row)
    else:
        remaining.append(row)

# New data: remaining first, then moved (preserve order)
new_data = remaining + moved

with open(IN, 'w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(header)
    writer.writerows(new_data)

print(f'Reordered {IN}: moved rows {MOVE_INDICES} to the end.')
