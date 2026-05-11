import csv
from decimal import Decimal, InvalidOperation

IN = r"result_summaries\summaries\new_synch\temp\summary_Qlearning_BF.csv"

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
    if 'gap_f_with_GVNS' not in fieldnames:
        fieldnames = fieldnames + ['gap_f_with_GVNS']

    # Compute gap for each row
    for r in rows:
        best_f = parse_decimal(r.get('f_method', ''))
        best_f_reimplementation = parse_decimal(r.get('f_GVNS', ''))
        gap = ''
        if best_f is not None and best_f_reimplementation is not None and best_f_reimplementation != 0:
            # gap = (best_f - best_f_reimplementation) / best_f_reimplementation * 100
            gap = (best_f - best_f_reimplementation) / best_f_reimplementation * 100
            # format with 2 decimal places
            gap = f"{gap.quantize(Decimal('0.01'))}"
        r['gap_f_with_GVNS'] = gap

    # Write back to the same file (overwrite)
    with open(IN, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {IN} with computed gap_f_with_GVNS values.")

if __name__ == '__main__':
    main()