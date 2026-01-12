import csv
from decimal import Decimal, InvalidOperation

IN = "result_summaries\merged_summary_qlearning_First_alpha01_gamma095.csv"

def parse_decimal(s):
    s = s.strip()
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None

def main():
    rows = []
    with open(IN, newline='') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for r in reader:
            rows.append(r)

    # Ensure the column exists
    if 'gap_f_with_paper' not in fieldnames:
        fieldnames = fieldnames + ['gap_f_with_paper']

    # Compute gap for each row
    for r in rows:
        best_f = parse_decimal(r.get('best_f', ''))
        best_f_paper = parse_decimal(r.get('best_f_paper', ''))
        gap = ''
        if best_f is not None and best_f_paper and best_f_paper != 0:
            # gap = (best_f - best_f_paper) / best_f_paper * 100
            gap = (best_f - best_f_paper) / best_f_paper * 100
            # format with 2 decimal places
            gap = f"{gap.quantize(Decimal('0.01'))}"
        r['gap_f_with_paper'] = gap

    # Write back to the same file (overwrite)
    with open(IN, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {IN} with computed gap_f_with_paper values.")

if __name__ == '__main__':
    main()
