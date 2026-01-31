#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'gvns' / 'ls_data' / 'FI'
pattern = '*_100_*.csv'

for p in sorted(root.glob(pattern)):
    with p.open('r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        print(f"Skipping empty file: {p}")
        continue
    header = rows[0]
    if 'gvns_iter' in header:
        print(f"Already has gvns_iter: {p}")
        continue
    if 'iter_gvns' not in header:
        print(f"No iter_gvns column in {p}, skipping")
        continue
    # insert gvns_iter as third column (index 2)
    new_header = header[:2] + ['gvns_iter'] + header[2:]
    iter_idx = header.index('iter_gvns')
    new_rows = [new_header]
    for row in rows[1:]:
        # ensure row length matches header length by padding
        row = row + [''] * (len(header) - len(row))
        gvns_val = row[iter_idx]
        new_row = row[:2] + [gvns_val] + row[2:]
        new_rows.append(new_row)
    # write back
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)
    print(f"Updated: {p}")
