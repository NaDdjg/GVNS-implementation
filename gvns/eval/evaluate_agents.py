import os
import csv
import time
from typing import List
from gvns.instance import parse_vrptws_txt
from gvns.local_search import paper_initial_fixed, gvns
from gvns.evaluate import evaluate_solution, is_feasible
from gvns.agents.ppo_agent import PPOAgent
from gvns.agents.qlearning_agent import QTableAgent
from gvns.agents.dqn_agent import DQNAgent

def load_agent(spec: dict):
    kind = spec.get("kind")
    if kind == "ppo":
        return PPOAgent(model_path=spec["will add path here"], vecnorm_path=spec.get("vecnorm_path"))
    elif kind == "qtable":
        ag = QTableAgent()
        ag.load(spec["will add path here"])
        return ag
    elif kind == "dqn":
        ag = DQNAgent(input_dim=spec["input_dim"], n_actions=spec["n_actions"], device=spec.get("device","cpu"))
        ag.load(spec["will addpath "])
        return ag
    elif kind == "baseline":
        return None
    else:
        raise ValueError("Unknown agent kind: " + str(kind))

def evaluate_on_instance(instance_path: str, agent, Tmax_seconds=300, verbose=False):
    inst = parse_vrptws_txt(instance_path)
    init_sol, _ = paper_initial_fixed(inst, num_vehicles_estimate=30, seed=42)
    t0 = time.time()
    sol, history = gvns(inst, Tmax_seconds=Tmax_seconds, agent=agent, verbose=verbose)
    elapsed = time.time() - t0
    stats = evaluate_solution(inst, sol)
    stats.update({
        "time": elapsed,
        "feasible": is_feasible(inst, sol, stats),
    })
    return sol, stats, history

def batch_evaluate(instance_list: List[str], agent_specs: List[dict], out_csv: str):
    # Build agent objects
    agents = [(spec.get("name", spec["kind"]), load_agent(spec)) for spec in agent_specs]

    rows = []
    for inst_path in instance_list:
        for name, agent in agents:
            sol, stats, hist = evaluate_on_instance(inst_path, agent)
            row = {
                "instance": os.path.basename(inst_path),
                "agent": name,
                "F": stats['F'],
                "f": stats['f'],
                "PenT": stats['PenT'],
                "PenH": stats['PenH'],
                "PenS": stats['PenS'],
                "time_s": stats['time'],
                "feasible": stats.get('feasible', False)
            }
            rows.append(row)
            print(f"Evaluated {inst_path} with {name}: F={row['F']:.3f} time={row['time_s']:.1f}s feasible={row['feasible']}")

    # write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["instance","agent","F","f","PenT","PenH","PenS","time_s","feasible"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved evaluation results to", out_csv)
