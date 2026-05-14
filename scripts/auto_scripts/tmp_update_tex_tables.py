from pathlib import Path
import re

files = [
    Path('result_summaries/papers/results_tables/contribution/dqn/cost/50_instances.tex'),
    Path('result_summaries/papers/results_tables/contribution/dqn/cost/100_instances.tex'),
    Path('result_summaries/papers/results_tables/contribution/dqn/cost/200_instances.tex'),
    Path('result_summaries/papers/results_tables/contribution/dqn/cost/400_instances.tex'),
]

for path in files:
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    out = []
    in_table = False
    header_done = False
    for line in lines:
        if line.startswith('\\begin{tabular}'):
            out.append('\\begin{tabular}{l r r r r @{\\hspace{8pt}} r r r r}')
            in_table = True
            continue
        if in_table and not header_done:
            if line.startswith('\\multirow{2}{*}{Instance}'):
                out.append('\\multirow{2}{*}{Instance} & \\multirow{2}{*}{Best} & \\multicolumn{3}{c}{GVNS} & \\multicolumn{4}{c}{DQN\\_FI} \\\\')
                continue
            if line.startswith('\\cmidrule(lr){3-5}'):
                out.append('\\cmidrule(lr){3-5} \\cmidrule(lr){6-9}')
                continue
            if line.strip() == '&  & Best & Avg & Time & Best & Avg & Time \\\\':
                out.append('&  & Best & Avg & Time & Best & Avg & Time & Gap \\\\')
                header_done = True
                continue
        if in_table and header_done and line.strip().startswith('\\texttt'):
            parts = [p.strip() for p in line.split('&')]
            if len(parts) == 8:
                instance, overall, dqn_best, dqn_avg, dqn_time, gvns_best, gvns_avg, gvns_time = parts
                gvns_time = gvns_time.rstrip(' \\\\')
                dqn_best_val = float(re.sub(r'[^0-9.-]', '', dqn_best))
                gvns_best_val = float(re.sub(r'[^0-9.-]', '', gvns_best))
                gap = dqn_best_val - gvns_best_val
                gap_str = f'{gap:.2f}'.replace('-0.00', '0.00')
                out.append(f'{instance} & {overall} & {gvns_best} & {gvns_avg} & {gvns_time} & {dqn_best} & {dqn_avg} & {dqn_time} & {gap_str} \\\\')
                continue
        if in_table and header_done and line.startswith('\\midrule'):
            out.append(line)
            continue
        if in_table and header_done and line.startswith('\\textbf{Average}'):
            parts = [p.strip() for p in line.split('&')]
            if len(parts) >= 4:
                overall = parts[1]
                dqn_best = parts[2]
                gvns_best = parts[5] if len(parts) > 5 else '-'
                out.append(f'\\textbf{{Average}} & {overall} & {gvns_best} & - & - & {dqn_best} & - & - & - \\\\')
                continue
        if line.startswith('\\end{tabular}'):
            in_table = False
            header_done = False
        out.append(line)
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('updated', path)