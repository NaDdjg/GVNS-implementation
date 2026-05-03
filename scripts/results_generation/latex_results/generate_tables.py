import os
import pandas as pd
from collections import defaultdict

# === PATHS ===
INPUT_DIR = r"result_summaries\summaries\new_synch"
OUTPUT_DIR = r"result_summaries\papers\results_tables\contribution\cost"
GAP_OUTPUT_DIR = r"result_summaries\papers\results_tables\contribution\gap"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GAP_OUTPUT_DIR, exist_ok=True)

# === DATA STRUCTURE ===
data = defaultdict(lambda: defaultdict(dict))
methods = set()

# === READ ALL CSV FILES ===
for file in os.listdir(INPUT_DIR):
    if not (file.endswith(".csv") and file.startswith("summary_")):
        continue

    method = file.replace("summary_", "").replace(".csv", "")
    methods.add(method)

    filepath = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(filepath)

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Detect needed columns safely
    f_cols = [c for c in df.columns if c.startswith("f_")]

    if not f_cols or not "avg_time" in df.columns or not "avg_f" in df.columns:
        print(f"Skipping {file} (missing required columns)")
        continue

    f_col = f_cols[0]

    # Drop empty rows
    df = df.dropna(subset=["instance"])

    for _, row in df.iterrows():
        instance = str(row["instance"]).strip()

        if not instance or instance == "nan":
            continue

        # === SIZE DETECTION ===
        parts = instance.split("_")

        try:
            second = int(parts[1])

            # Small instances
            if second in [50, 100]:
                size = second
            else:
                # Large instances (encoded)
                size = second * 100
        except:
            continue

        # === VALUES ===
        try:
            f_val = float(row[f_col])
            time_val = float(row["avg_time"])
            avg_val = float(row["avg_f"])
        except:
            continue

        data[size][instance][method] = {
            "f": f_val,
            "avg": avg_val,
            "time": time_val
        }

methods = sorted(methods)

# === GENERATE LATEX TABLES ===
for size, instances in data.items():

    lines = []

    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append(f"\\caption{{Results for {size}-customer instances}}")
    lines.append("\\vspace{4mm}")

    # Column format
    col_format = "l r"
    for _ in methods:
        col_format += " r r r @{\hspace{8pt}}"

    # Resize to fit page
    lines.append("\\begin{adjustbox}{width=\\textwidth}")
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")

    # === HEADER ===
    header1 = ["\\multirow{2}{*}{Instance}", "\\multirow{2}{*}{Best}"]

    for m in methods:
        safe_m = m.replace("_", "\\_")
        header1.append(f"\\multicolumn{{3}}{{c}}{{{safe_m}}}")

    lines.append(" & ".join(header1) + " \\\\")

    # cmidrules
    cmid = []
    start = 3
    for _ in methods:
        cmid.append(f"\\cmidrule(lr){{{start}-{start+2}}}")
        start += 3

    lines.append(" ".join(cmid))

    # Sub-header
    header2 = ["", ""]
    for _ in methods:
        header2 += ["Best", "Avg", "Avg_time"]

    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    # === AVERAGE ACCUMULATORS ===
    best_values_all = []
    method_values_all = {m: [] for m in methods}

    # === ROWS ===
    for instance, methods_data in sorted(instances.items()):

        # Escape + format instance name
        safe_instance = "\\texttt{" + instance.replace("_", "\\_") + "}"

        # Get f values
        f_values = {
            m: methods_data[m]["f"]
            for m in methods
            if m in methods_data
        }

        if not f_values:
            continue

        best_overall = min(f_values.values())
        best_values_all.append(best_overall)

        row = [safe_instance, f"{best_overall:.2f}"]

        for m in methods:
            if m in methods_data:
                f_val = methods_data[m]["f"]
                avg_val = methods_data[m]["avg"]
                avg_time = methods_data[m]["avg_time"]

                method_values_all[m].append(f_val)

                # Bold best method
                if abs(f_val - best_overall) < 1e-6:
                    f_str = f"\\textbf{{{f_val:.2f}}}"
                else:
                    f_str = f"{f_val:.2f}"

                row.append(f_str)
                row.append(f"{avg_val:.2f}")
                row.append(f"{avg_time:.1f}")
            else:
                row += ["-", "-", "-"]

        lines.append(" & ".join(row) + " \\\\")

    # === AVERAGE ROW ===
    lines.append("\\midrule")

    avg_best = sum(best_values_all) / len(best_values_all) if best_values_all else 0

    avg_row = ["\\textbf{Average}", f"{avg_best:.2f}"]

    for m in methods:
        if method_values_all[m]:
            avg_f = sum(method_values_all[m]) / len(method_values_all[m])
            avg_row.append(f"{avg_f:.2f}")
            avg_row.append("-")  # no avg of avg
            avg_row.append("-")  # no time
        else:
            avg_row += ["-", "-", "-"]

    lines.append(" & ".join(avg_row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")

    # === WRITE FILE ===
    output_file = os.path.join(OUTPUT_DIR, f"{size}_instances.tex")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

print("Tables generated successfully!")

# === GENERATE GAP TABLES ===
for size, instances in data.items():

    lines = []

    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append(f"\\caption{{Gap results for {size}-customer instances}}")
    lines.append("\\vspace{4mm}")

    # Column format (only 2 columns per method now: gap + time)
    col_format = "l r"
    for _ in methods:
        col_format += " r r @{\hspace{8pt}}"

    lines.append("\\begin{adjustbox}{width=\\textwidth}")
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")

    # === HEADER ===
    header1 = ["\\multirow{2}{*}{Instance}", "\\multirow{2}{*}{Best}"]

    for m in methods:
        safe_m = m.replace("_", "\\_")
        header1.append(f"\\multicolumn{{2}}{{c}}{{{safe_m}}}")

    lines.append(" & ".join(header1) + " \\\\")

    # cmidrules
    cmid = []
    start = 3
    for _ in methods:
        cmid.append(f"\\cmidrule(lr){{{start}-{start+1}}}")
        start += 2

    lines.append(" ".join(cmid))

    # Sub-header
    header2 = ["", ""]
    for _ in methods:
        header2 += ["Gap(\\%)", "Time"]

    lines.append(" & ".join(header2) + " \\\\")
    lines.append("\\midrule")

    # === ACCUMULATORS ===
    gap_values_all = {m: [] for m in methods}

    # === ROWS ===
    for instance, methods_data in sorted(instances.items()):

        safe_instance = "\\texttt{" + instance.replace("_", "\\_") + "}"

        f_values = {
            m: methods_data[m]["f"]
            for m in methods
            if m in methods_data
        }

        if not f_values:
            continue

        best_overall = min(f_values.values())

        row = [safe_instance, f"{best_overall:.2f}"]

        for m in methods:
            if m in methods_data:
                f_val = methods_data[m]["f"]
                avg_time = methods_data[m]["avg_time"]

                # GAP calculation
                gap = 100 * (f_val - best_overall) / best_overall

                gap_values_all[m].append(gap)

                # Bold best (gap = 0)
                if abs(gap) < 1e-6:
                    gap_str = "\\textbf{0.00}"
                else:
                    gap_str = f"{gap:.2f}"

                row.append(gap_str)
                row.append(f"{time_val:.1f}")
            else:
                row += ["-", "-"]

        lines.append(" & ".join(row) + " \\\\")

    # === AVERAGE ROW ===
    lines.append("\\midrule")

    avg_row = ["\\textbf{Average}", "-"]

    for m in methods:
        if gap_values_all[m]:
            avg_gap = sum(gap_values_all[m]) / len(gap_values_all[m])
            avg_row.append(f"{avg_gap:.2f}")
            avg_row.append("-")
        else:
            avg_row += ["-", "-"]

    lines.append(" & ".join(avg_row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")

    # === WRITE FILE ===
    output_file = os.path.join(GAP_OUTPUT_DIR, f"{size}_instances.tex")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

print("Gap tables generated successfully!")