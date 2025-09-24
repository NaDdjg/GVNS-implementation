import os, glob, time
import pandas as pd
import matplotlib.pyplot as plt
from .instance import parse_vrptws_txt, Instance, SEED
from .local_search import gvns
from .evaluate import evaluate_solution
from .solution import Solution
from .utils import infer_vehicle_estimate

def run_multiple(inst, runs=5, Tmax_seconds=10, **gvns_kwargs):
    results = []
    for i in range(runs):
        seed = SEED + i*7
        start_time = time.time()
        best, history = gvns(inst, Tmax_seconds=Tmax_seconds, seed=seed, **gvns_kwargs)
        t_run = time.time() - start_time
        eval_best = evaluate_solution(inst, best, gvns_kwargs.get('w1_init', 0.5),
                                      gvns_kwargs.get('w2_init', 0.1), gvns_kwargs.get('M', 1e6))
        results.append({'run': i, 'best_F': eval_best['F'], 'f': eval_best['f'],
                        'PenT': eval_best['PenT'], 'PenH': eval_best['PenH'], 'PenS': eval_best['PenS'],
                        'time_s': t_run})
        print(f"Run {i}: best_F={eval_best['F']:.3f}, time={t_run:.1f}s")
    df = pd.DataFrame(results)
    return df

def solution_details(inst: Instance, sol: Solution):
    # return per-visit info
    details = []
    for r_idx, route in enumerate(sol.routes):
        cur_time = 0.0
        cur_node = inst.depot
        for v_id in route:
            travel = inst.travel_time[(cur_node.id, v_id)]
            arrival = cur_time + travel
            visit = inst.visits[v_id]
            start = max(arrival, visit.tw_early)
            depart = start + visit.service
            details.append({'visit': v_id, 'route': r_idx, 'arrival': arrival, 'start': start, 'late': max(0.0, start - visit.tw_late)})
            cur_time = depart
            cur_node = visit
    return pd.DataFrame(details)

def benchmark_folder(folder='vrptws',
                     runs_per_instance=5,
                     Tmax_rule=lambda n: None,
                     gvns_kwargs=None,
                     out_csv='gvns_benchmark_results.csv'):

    if gvns_kwargs is None:
        gvns_kwargs = {
            'contmax':100, 'p_init':1, 'p_max':10, 'Dp':0.3,
            'w1_init':0.5, 'w2_init':0.1, 'alpha':0.1,
            'delta':0.4, 'M':1e6, 'w1_max':None, 'w2_max':None,
            'update_every':1
        }

    results_dir = os.path.join(os.getcwd(), "results")
    plots_dir = os.path.join(os.getcwd(), "results/plots")
    traces_dir = os.path.join(os.getcwd(), "results/traces")
    os.makedirs(results_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, '**', '*.txt'), recursive=True))
    
    if not files:
        raise FileNotFoundError(f"No .txt files found in {os.path.abspath(folder)}")
    
    results = []

    for fpath in files:
        instance_name = os.path.basename(fpath)
        instance_class = os.path.basename(os.path.dirname(fpath))
        num_veh_est = infer_vehicle_estimate()

        print(f"Benchmarking: {fpath} (class={instance_class})")

        inst = parse_vrptws_txt(fpath)
        n = len(inst.visits)
        Tmax = Tmax_rule(n)
        per_instance_values = []

        for run_id in range(runs_per_instance):
            seed = SEED + run_id*31
            t0 = time.time()
            best, history = gvns(
                inst, 
                Tmax_seconds=Tmax, 
                seed=seed, 
                verbose=False, 
                num_vehicles_estimate=num_veh_est,
                **gvns_kwargs)
            t_run = time.time() - t0

            # inspect convergence
            times  = [h['time']    for h in history]
            f_best = [h['f_best']  for h in history]
            f_hist  = [h['raw_f']  for h in history]

            print(f"    improvement trace (f over time): {f_hist}")

            plt.figure()
            plt.plot(times, f_hist, marker='.')
            plt.xlabel("time (s)")
            plt.ylabel("f")
            plt.title(f"Convergence – {instance_name} run {run_id}")
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir,
                        f"curve_{instance_name}_run{run_id}_f_improvement.png"))
            plt.close()

            plt.figure()
            plt.plot(times, f_best, marker='.')
            plt.xlabel("time (s)")
            plt.ylabel("best f")
            plt.title(f"Convergence – {instance_name} run {run_id}")
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir,
                        f"curve_{instance_name}_run{run_id}_best_f.png"))
            plt.close()

            pd.DataFrame(history).to_csv(
                os.path.join(traces_dir,
                    f"trace_{instance_name}_run{run_id}.csv"),
                index=False)

            if history:
                min_F = min(h['F_best'] for h in history)
                # first time we hit that min
                best_time = min(h['time'] for h in history if abs(h['F_best']-min_F) < 1e-9)
            else:
                best_time = t_run  # fallback if history empty

            stats = evaluate_solution(inst, best,
                                      gvns_kwargs.get('w1_init',0.5),
                                      gvns_kwargs.get('w2_init',0.1),
                                      gvns_kwargs.get('M',None))
            per_instance_values.append({
                'instance': instance_name,
                'class': instance_class,
                'run': run_id,
                'best_F': stats['F'],
                'f': stats['f'],
                'PenT': stats['PenT'],
                'PenH': stats['PenH'],
                'PenS': stats['PenS'],
                'time_s': t_run,
                'best_time_s': best_time       
            })
            print(f"  run {run_id}: best_F={stats['F']:.3f}, "
                  f"f={stats['f']:.3f}, PenT={stats['PenT']:.3f}, "
                  f"time={t_run:.1f}s, best_time={best_time:.1f}s")

        # save per-run details
        run_csv = os.path.join(results_dir, f"results_{instance_name}.csv")
        pd.DataFrame(per_instance_values).to_csv(run_csv, index=False)

        dfpi = pd.DataFrame(per_instance_values)

        # per-instance summary: use mean best_time across runs
        summary = {
            'instance': instance_name,
            'class': instance_class,
            'best_over_runs': dfpi['best_F'].min(),
            'avg_over_runs': dfpi['best_F'].mean(),
            'std_over_runs': dfpi['best_F'].std(),
            'avg_time': dfpi['time_s'].mean(),
            'avg_best_time': dfpi['best_time_s'].mean(), 
            'n_customers': n
        }
        results.append(summary)

    dfres = pd.DataFrame(results)
    out_csv_path = os.path.join(results_dir, out_csv)
    dfres.to_csv(out_csv_path, index=False)
    print("Benchmark finished. Results saved to", out_csv)
    return dfres

def get_instance_class(instance_name):
    base = instance_name.upper().replace('.TXT','')
    if base.startswith('RC'):
        return 'RC' + base[2]
    elif base.startswith('R'):
        return 'R' + base[1]
    elif base.startswith('C'):
        return 'C' + base[1]
    else:
        return 'Unknown'
    
def summarize_by_class(df_summary):
    grouped = df_summary.groupby('class').agg({
        'best_over_runs': ['mean','std'],
        'avg_over_runs': 'mean',
        'n_customers': 'mean',
        'avg_time': 'mean'
    }).reset_index()

    grouped.columns = ['class','best_mean','best_std','avg_mean','n_customers','avg_time']
    return grouped
