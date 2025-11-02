# gvns/gvns.py
import time
import random
from typing import Optional, Dict, Any

from gvns.solution import Solution
from gvns.initializer import random_insertion
import gvns.operators as ops
import gvns.local_search as ls

DEFAULT_PARAMS = {
    "Contmax": 100,
    "p_init": 1,
    "p_max": 10,
    "Dp": 0.3,
    "w1": 0.5,
    "w2": 0.1,
    "alpha": 0.1,
    "delta": 0.4,
    "M": 1e6,
    "seed": 0,
    "Tmax": None,
}

def run_gvns(instance, params=None, verbose=True, trace_list=None):
    if params is None:
        cfg = DEFAULT_PARAMS.copy()
    else:
        cfg = params
    random.seed(cfg["seed"])

    # compute expected total visits (accounts for sync extra occurrences)
    depot_id = min(instance.nodes.keys())
    all_non_depot = [nid for nid in instance.nodes.keys() if nid != depot_id]
    # Each non-sync customer appears once; each sync customer appears req times (often 2)
    expected_total = len([c for c in all_non_depot if c not in instance.sync_customers]) + \
                     sum(instance.sync_requirements.get(c,1) for c in instance.sync_customers)

    def total_visits(sol: Solution):
        return sum(len(rt) for rt in sol.tours)

    def occ_distribution(sol: Solution):
        from collections import Counter
        cnt = Counter()
        for rt in sol.tours:
            for v in rt:
                cnt[v] += 1
        return cnt

    # init
    current = random_insertion(instance, seed=cfg["seed"])
    current.w1 = cfg["w1"]; current.w2 = cfg["w2"]; current.alpha = cfg["alpha"]; current.M = cfg["M"]
    # quick validator of initializer:
    tv_init = total_visits(current)
    if tv_init != expected_total:
        print("[GVNS][ERROR] initializer built unexpected total_visits:", tv_init, "expected:", expected_total)
        print(" occurrences sample:", dict(list(occ_distribution(current).items())[:20]))

    current.eval()
    best = current.copy()
    best.eval()

    p = cfg["p_init"]
    cont = 0
    iteration_num = 0
    start_time = time.time()
    time_limit = cfg["Tmax"]
    delta = cfg["delta"]
    p_min = cfg["p_init"]; p_max = cfg["p_max"]
    Dp = cfg["Dp"]

    if verbose:
        print(f"[GVNS] start: F={current.Fval:.4f}, f(travel)={current.travel_time:.4f}, PenT={current.penT:.4f}, PenH={current.penH:.4f}, PenS={current.penS}")

    while True:
        iteration_num += 1
        now = time.time()
        elapsed = now - start_time
        if time_limit and elapsed >= time_limit:
            if verbose: print("[GVNS] time limit reached.")
            break
        if cont >= cfg["Contmax"]:
            if verbose: print("[GVNS] cont limit reached.")
            break

        # Shake
        k = max(1, int(p))
        shaken = ops.random_shake(current, k=k, insert_prob=Dp)
        # keep penalty weights
        shaken.w1 = current.w1; shaken.w2 = current.w2; shaken.alpha = current.alpha; shaken.M = current.M

        # Local search (B-VND)
        improved = ls.b_vnd(shaken, neighborhoods=["N1","N2","N3","N4"], max_no_improve_iters=10, time_limit=5.0)
        improved.eval()

        new = improved

        # Defensive checks: total visits, valid node ids, non-zero F
        tv_new = total_visits(new)
        if tv_new != expected_total:
            print(f"[GVNS][iteration {iteration_num}] REJECT candidate - total_visits {tv_new} != expected {expected_total}")
            print(" occ distribution (sample):", dict(list(occ_distribution(new).items())[:30]))
            cont += 1
            # adjust shaking intensity to avoid repeated invalid perturbations
            p = max(p_min, p - 1)
            # print iteration summary anyway
            print(f"[GVNS][iteration {iteration_num}] current F={current.Fval:.4f} f={current.travel_time:.4f} best F={best.Fval:.4f} elapsed={elapsed:.2f}s cont={cont} p={p}")
            continue

        # reject extreme zero-valued objective (likely corruption)
        if new.Fval == 0.0 or new.travel_time == 0.0 and new.penS == 0.0 and new.penT == 0.0 and new.penH == 0.0:
            print(f"[GVNS][iteration {iteration_num}] REJECT candidate - zero objective suspicious (F=0). Occ dist sample: {dict(list(occ_distribution(new).items())[:30])}")
            cont += 1
            p = max(p_min, p - 1)
            print(f"[GVNS][iteration {iteration_num}] current F={current.Fval:.4f} f={current.travel_time:.4f} best F={best.Fval:.4f} elapsed={elapsed:.2f}s cont={cont} p={p}")
            continue

        # Acceptance (Move-or-Not)
        threshold = (1.0 + delta) * current.Fval
        accept = (new.Fval <= threshold)

        if accept:
            current = new
            cont = 0
            accepted_str = "ACCEPTED"
        else:
            cont += 1
            accepted_str = "REJECTED"

        # Update best
        if (
            current.Fval < best.Fval
            and current.Fval > 1e-6
            and all(len(t) > 0 for t in new.tours)
        ):
            best = new.copy()
            print(
                f"[GVNS][iteration {iteration_num}] NEW BEST: "
                f"F={best.Fval:.4f} f={best.travel_time:.4f} "
                f"PenT={best.penT:.4f} PenH={best.penH:.4f} PenS={best.penS:.1f} "
                f"elapsed={elapsed:.2f}s"
            )

        # Penalty self-adjust (Algorithm 6)
        current.eval()
        current.self_adjust_penalties()

        # update p (simple schedule: increase if stagnating, reduce on accept)
        if cont > 0 and cont % 10 == 0:
            p = min(p + 1, p_max)
        if accept:
            p = max(p_min, p - 1)

        # Print per-iteration summary (you requested every iteration)
        print(f"[GVNS][iteration {iteration_num}] {accepted_str} | F={new.Fval:.4f} f={new.travel_time:.4f} PenT={new.penT:.4f} PenH={new.penH:.4f} PenS={new.penS} | current_F={current.Fval:.4f} best_F={best.Fval:.4f} cont={cont} p={p} elapsed={elapsed:.2f}s")
        if trace_list is not None:
            trace_list.append([
                iteration_num,
                current.Fval,
                current.travel_time,
                current.penT,
                current.penH,
                current.penS,
                elapsed
            ])
    # final
    best.eval()
    return best
