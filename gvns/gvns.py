import time
import random
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

    depot_id = min(instance.nodes.keys())
    all_non_depot = [nid for nid in instance.nodes.keys() if nid != depot_id]
    expected_total = len([c for c in all_non_depot if c not in instance.sync_customers]) + \
                     sum(instance.sync_requirements.get(c, 1) for c in instance.sync_customers)

    def total_visits(sol: Solution):
        return sum(len(rt) for rt in sol.tours)

    def occ_distribution(sol: Solution):
        from collections import Counter
        cnt = Counter()
        for rt in sol.tours:
            for v in rt:
                cnt[v] += 1
        return cnt

    # initialization 
    current = random_insertion(instance, seed=cfg["seed"])
    current.w1 = cfg["w1"]; current.w2 = cfg["w2"]; current.alpha = cfg["alpha"]; current.M = cfg["M"]

    tv_init = total_visits(current)
    if tv_init != expected_total:
        print("[GVNS][ERROR] initializer built unexpected total_visits:", tv_init, "expected:", expected_total)
        print(" occurrences sample:", dict(list(occ_distribution(current).items())[:20]))

    current.eval()

    best_aug = current.copy()
    best_aug.eval()

    best_feasible = None
    best_feasible_f = float("inf")
    if current.penT == 0.0 and current.penH == 0.0 and current.penS == 0.0:
        best_feasible = current.copy()
        best_feasible_f = current.travel_time

    p = cfg["p_init"]
    cont = 0
    iteration_num = 0
    start_time = time.time()
    time_limit = cfg["Tmax"]
    delta = cfg["delta"]
    p_min, p_max = cfg["p_init"], cfg["p_max"]
    Dp = cfg["Dp"]

    if verbose:
        print(f"[GVNS] start: F={current.Fval:.4f}, f(travel)={current.travel_time:.4f}, "
              f"PenT={current.penT:.4f}, PenH={current.penH:.4f}, PenS={current.penS}")

    def compute_bvnd_time(n, sync_pairs, m,
                          base_per_customer=0.4,
                          per_sync_pair=4.0,
                          per_vehicle=0.05,
                          min_time=10.0,
                          max_time=600.0,
                          Tmax=None,
                          frac_of_Tmax=0.8):
        t = base_per_customer * n + per_sync_pair * sync_pairs + per_vehicle * m
        if Tmax is not None:
            t = min(t, frac_of_Tmax * Tmax)
        t = max(min_time, min(t, max_time))
        return float(t)

    # main loop
    while True:
        iteration_num += 1
        elapsed = time.time() - start_time
        if time_limit and elapsed >= time_limit:
            if verbose: print("[GVNS] time limit reached.")
            break
        if cont >= cfg["Contmax"]:
            if verbose: print("[GVNS] cont limit reached.")
            break

        # shake
        k = max(1, int(p))
        shaken = ops.random_shake(current, k=k, insert_prob=Dp)
        shaken.w1 = current.w1; shaken.w2 = current.w2; shaken.alpha = current.alpha; shaken.M = current.M

        # local search (B-VND)
        n_customers = len([nid for nid in instance.nodes if nid != depot_id])
        sync_pairs = len(getattr(instance, "sync_customers", []))
        m = instance.m
        bvnd_time = compute_bvnd_time(n=n_customers, sync_pairs=sync_pairs, m=m, Tmax=time_limit)

        improved = ls.b_vnd(
            shaken,
            neighborhoods=["N1", "N2", "N3", "N4"],
            max_no_improve_iters=10,
            time_limit=bvnd_time
        )
        improved.eval()
        new = improved

        # defensive checks
        tv_new = total_visits(new)
        if tv_new != expected_total or (new.Fval == 0.0 and (new.penT + new.penH + new.penS) > 0):
            cont += 1
            p = max(p_min, p - 1)
            if trace_list is not None:
                trace_list.append([
                    iteration_num, current.Fval, current.travel_time,
                    current.penT, current.penH, current.penS, elapsed, best_feasible_f
                ])
            continue

        # acceptance (Move-or-Not)
        threshold = (1.0 + delta) * current.Fval
        accept = (new.Fval <= threshold)
        if accept:
            current = new
            cont = 0
            accepted_str = "ACCEPTED"
        else:
            cont += 1
            accepted_str = "REJECTED"

        # update augmented best (penalized objective) 
        if new.Fval < best_aug.Fval and new.Fval > 1e-6:
            best_aug = new.copy()
            print(f"[GVNS][iteration {iteration_num}] NEW BEST (augmented): "
                  f"F={best_aug.Fval:.4f} f={best_aug.travel_time:.4f} "
                  f"PenT={best_aug.penT:.4f} PenH={best_aug.penH:.4f} PenS={best_aug.penS:.1f} "
                  f"elapsed={elapsed:.2f}s")

        # update feasible best (pure f) 
        if new.penT == 0.0 and new.penH == 0.0 and new.penS == 0.0 and new.travel_time < best_feasible_f:
            best_feasible = new.copy()
            best_feasible_f = new.travel_time
            print(f"[GVNS][iteration {iteration_num}] NEW BEST (feasible f): "
                  f"f={best_feasible_f:.4f} elapsed={elapsed:.2f}s")

        # penalty adaptation
        current.eval()
        current.self_adjust_penalties()

        # update p
        if cont > 0 and cont % 10 == 0:
            p = min(p + 1, p_max)
        if accept:
            p = max(p_min, p - 1)

        # iteration summary
        print(f"[GVNS][iteration {iteration_num}] {accepted_str} | "
              f"F={new.Fval:.4f} f={new.travel_time:.4f} PenT={new.penT:.4f} "
              f"PenH={new.penH:.4f} PenS={new.penS} | "
              f"current_F={current.Fval:.4f} best_F={best_aug.Fval:.4f} "
              f"cont={cont} p={p} elapsed={elapsed:.2f}s")

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

    best_aug.eval()
    return best_aug