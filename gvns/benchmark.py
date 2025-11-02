import os, glob, time
import pandas as pd
import matplotlib.pyplot as plt
from .instance import parse_vrptws_txt, Instance, SEED
from .local_search import gvns
from .evaluate import evaluate_solution
from .solution import Solution
from .utils import infer_vehicle_estimate 

# this is to run tests when i generated the instances (not the benchmark)
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
                     out_csv='gvns_benchmark_results.csv',
                     agent_factory=None):

    if gvns_kwargs is None:
        gvns_kwargs = {
            'contmax':100, 'p_init':1, 'p_max':10, 'Dp':0.3,
            'w1_init':0.5, 'w2_init':0.1, 'alpha':0.1,
            'delta':0.4, 'M':1e6, 'w1_max':None, 'w2_max':None,
            'update_every':1
        }

    cwd = os.getcwd()
    results_dir = os.path.join(cwd, "results")
    plots_dir = os.path.join(results_dir, "plots")
    traces_dir = os.path.join(results_dir, "traces")
    models_dir = os.path.join(cwd, "models")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if agent_factory is not None:
        try:
            agent_name = getattr(agent_factory, "__name__", None) or "agent"
            if agent_name in ("<lambda>", "partial", None):
                # fallback: make a temporary agent to read its class name
                tmp_agent = agent_factory()
                agent_name = tmp_agent.__class__.__name__
                del tmp_agent
        except Exception:
            agent_name = "agent"
    else:
        agent_name = "classic"

    agent_results_dir = os.path.join(results_dir, "agents", agent_name)
    agent_plots_dir = os.path.join(agent_results_dir, "plots")
    agent_traces_dir = os.path.join(agent_results_dir, "traces")
    agent_models_dir = os.path.join(models_dir, agent_name)

    os.makedirs(agent_results_dir, exist_ok=True)
    os.makedirs(agent_plots_dir, exist_ok=True)
    os.makedirs(agent_traces_dir, exist_ok=True)
    os.makedirs(agent_models_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, '**', '*.txt'), recursive=True))
    
    if not files:
        raise FileNotFoundError(f"No .txt files found in {os.path.abspath(folder)}")
    
    results = []

    for fpath in files:
        instance_name = os.path.basename(fpath)
        instance_class = os.path.basename(os.path.dirname(fpath))

        print(f"Benchmarking: {fpath} (class={instance_class})")

        inst = parse_vrptws_txt(fpath)
        num_veh_est = infer_vehicle_estimate()

        if inst.sync_pairs:
            print(f"  sync_pairs found in file: {len(inst.sync_pairs)}")
        else:
            print(f"  no sync pairs for this instance")

        n = len(inst.visits)
        Tmax = Tmax_rule(n)
        per_instance_values = []

        for run_id in range(runs_per_instance):
            seed = SEED + run_id*31
            t0 = time.time()

            print(f"number of vehicles: {num_veh_est}")

            agent = agent_factory() if agent_factory is not None else None
            best, history = gvns(
                inst, 
                Tmax_seconds=Tmax, 
                seed=seed, 
                verbose=False, 
                num_vehicles_estimate=num_veh_est,
                agent=agent,
                **gvns_kwargs)
            t_run = time.time() - t0

            if agent is not None:
                agent_path = os.path.join(agent_models_dir, f"QTable_{instance_name}_run{run_id}.pkl")
                os.makedirs(os.path.dirname(agent_path), exist_ok=True)
                try:
                    agent.save(agent_path)
                except Exception as e:
                    print(f"[WARN] could not save agent model: {e}")

                # also save a CSV summary of the Q-table if available
                try:
                    if hasattr(agent, "Q"):
                        q_items = []
                        for k, v in agent.Q.items():
                            row = {"state": repr(k)}
                            for i, val in enumerate(v):
                                row[f"q{i}"] = float(val)
                            q_items.append(row)
                        if q_items:
                            qdf = pd.DataFrame(q_items)
                            qcsv = os.path.join(agent_traces_dir, f"qtable_{instance_name}_run{run_id}.csv")
                            qdf.to_csv(qcsv, index=False)
                except Exception as e:
                    print(f"[WARN] could not dump Q-table CSV: {e}")

            # inspect convergence
            times = [h.get('time', None) for h in history]
            raw_f = [h.get('raw_f', None) for h in history]
            best_f = [h.get('f_best', None) for h in history]

            # plots
            if agent is not None:
                try:
                    plt.figure()
                    plt.plot(times, raw_f, marker='.', label='raw_f')
                    plt.plot(times, best_f, linewidth=2.0, linestyle='-', marker=None, label='best_f')
                    plt.xlabel("time (s)")
                    plt.ylabel("f")
                    plt.title(f"Convergence – {instance_name} run {run_id} ({agent_name})")
                    plt.grid(True)
                    plt.legend()
                    afname = os.path.join(agent_plots_dir, f"curve_{instance_name}_run{run_id}_combined_{agent_name}.png")
                    plt.savefig(afname)
                    plt.close()
                except Exception as e:
                    print(f"[WARN] plotting agent-specific failed: {e}")
            else:
                try:
                    plt.figure()
                    plt.plot(times, raw_f, marker='.', label='raw_f')
                    plt.plot(times, best_f, linewidth=2.0, linestyle='-', marker=None, label='best_f')
                    plt.xlabel("time (s)")
                    plt.ylabel("f")
                    plt.title(f"Convergence – {instance_name} run {run_id}")
                    plt.grid(True)
                    plt.legend()
                    fname = os.path.join(plots_dir, f"curve_{instance_name}_run{run_id}_combined.png")
                    plt.savefig(fname)
                    plt.close()
                except Exception as e:
                    print(f"[WARN] plotting classic combined failed: {e}")

            if agent is not None:
                try:
                    trace_agent_csv = os.path.join(agent_traces_dir, f"trace_{instance_name}_run{run_id}_{agent_name}.csv")
                    pd.DataFrame(history).to_csv(trace_agent_csv, index=False)
                except Exception as e:
                    print(f"[WARN] saving agent trace failed: {e}")
            else:
                try:
                    trace_csv = os.path.join(traces_dir, f"trace_{instance_name}_run{run_id}.csv")
                    pd.DataFrame(history).to_csv(trace_csv, index=False)
                except Exception as e:
                    print(f"[WARN] saving classic trace failed: {e}")

            if history:
                try:
                    min_f = min(h['f_best'] for h in history)
                    best_time = min(h['time'] for h in history if abs(h['f_best']-min_f) < 1e-9)
                except Exception:
                    best_time = t_run
            else:
                best_time = t_run

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

# summarize results by class
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
