import argparse
import re
from collections import defaultdict

CAPTION_SIZE_RE = re.compile(r"Enhanced GVNS results for (\d+)-customer instances")
ROW_RE = re.compile(r"\\texttt\{([^}]+)\}\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*\\\\")


def instance_size(name: str) -> int:
    m = re.search(r"_(50|100|200|400)(?:_|$)", name)
    if m:
        return int(m.group(1))
    return 0


def instance_type(name: str) -> str:
    if name.startswith("RC"):
        return "RC"
    if name.startswith("R"):
        return "R"
    if name.startswith("C"):
        return "C"
    return "?"


def parse_rows(tex_path: str):
    rows = []
    current_size = 0
    with open(tex_path, "r", encoding="utf-8") as f:
        for line in f:
            caption_match = CAPTION_SIZE_RE.search(line)
            if caption_match:
                current_size = int(caption_match.group(1))
                continue

            m = ROW_RE.search(line)
            if m:
                name, best, time, avg_cost, avg_time = m.groups()
                size = current_size or instance_size(name)
                rows.append({
                    "name": name,
                    "type": instance_type(name),
                    "size": size,
                    "best": float(best),
                    "time": float(time),
                    "avg_cost": float(avg_cost),
                    "avg_time": float(avg_time),
                })
    return rows


def compute_summaries(rows):
    size_groups = defaultdict(lambda: {"best": 0.0, "avg_cost": 0.0, "time": 0.0, "count": 0})
    type_groups = defaultdict(lambda: {"best": 0.0, "count": 0})

    for row in rows:
        size = row["size"]
        if size not in (50, 100, 200, 400):
            continue
        size_groups[size]["best"] += row["best"]
        size_groups[size]["avg_cost"] += row["avg_cost"]
        size_groups[size]["time"] += row["time"]
        size_groups[size]["count"] += 1

        type_key = (size, row["type"])
        type_groups[type_key]["best"] += row["best"]
        type_groups[type_key]["count"] += 1

    size_summary = {}
    for size, values in sorted(size_groups.items()):
        count = values["count"]
        if count == 0:
            continue
        size_summary[size] = {
            "avg_best_cost": values["best"] / count,
            "avg_mean_cost": values["avg_cost"] / count,
            "avg_runtime": values["time"] / count,
        }

    type_summary = defaultdict(dict)
    for (size, instance_type), values in sorted(type_groups.items()):
        if values["count"] == 0:
            continue
        type_summary[size][instance_type] = values["best"] / values["count"]

    return size_summary, type_summary


def format_number(value: float) -> str:
    return f"{value:.2f}"


def latex_summary_tables(size_summary, type_summary):
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\caption{Average performance of the enhanced GVNS on the generated benchmark instances}")
    lines.append("\\vspace{4mm}")
    lines.append("\\begin{adjustbox}{width=\\textwidth}")
    lines.append("\\begin{tabular}{c c c c}")
    lines.append("\\toprule")
    lines.append("Size & Avg Best Cost & Avg Mean Cost & Avg Runtime ")
    lines.append("\\midrule")
    for size in sorted(size_summary.keys()):
        row = size_summary[size]
        lines.append(f"{size} & {format_number(row['avg_best_cost'])} & {format_number(row['avg_mean_cost'])} & {format_number(row['avg_runtime'])} ")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\caption{Average best cost of the enhanced GVNS by instance type}")
    lines.append("\\vspace{4mm}")
    lines.append("\\begin{adjustbox}{width=\\textwidth}")
    lines.append("\\begin{tabular}{c c c c}")
    lines.append("\\toprule")
    lines.append("Size & C & R & RC")
    lines.append("\\midrule")
    for size in sorted(type_summary.keys()):
        c = format_number(type_summary[size].get("C", 0.0)) if "C" in type_summary[size] else "-"
        r = format_number(type_summary[size].get("R", 0.0)) if "R" in type_summary[size] else "-"
        rc = format_number(type_summary[size].get("RC", 0.0)) if "RC" in type_summary[size] else "-"
        lines.append(f"{size} & {c} & {r} & {rc} ")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX summary tables from an enhanced GVNS LaTeX result file.")
    parser.add_argument("tex_file", nargs="?", default="result_summaries/papers/results_tables/literature/summary_GVNS.tex", help="Path to the LaTeX source file to parse")
    parser.add_argument("--write", action="store_true", help="Write summary tables back into the LaTeX file before \\\"\\end{document}\\\"")
    args = parser.parse_args()

    rows = parse_rows(args.tex_file)
    size_summary, type_summary = compute_summaries(rows)
    result = latex_summary_tables(size_summary, type_summary)

    if args.write:
        with open(args.tex_file, "r", encoding="utf-8") as f:
            text = f.read()
        if "\\end{document}" not in text:
            raise RuntimeError("Could not find \\end{document} marker in file")
        text = text.replace("\\end{document}", result + "\n\\end{document}")
        with open(args.tex_file, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(result)


if __name__ == "__main__":
    main()
