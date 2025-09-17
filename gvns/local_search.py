from typing import Tuple
from .solution import Solution
from .instance import Instance
from .operators import (intra_route_insertion, inter_route_insertion,
                        intra_route_swap, inter_route_swap, clone_solution, shake_solution)
from .evaluate import evaluate_solution, quick_repair_greedy, is_feasible
from .viz import visualize_solution
import random, time
from .instance import random_insertion, SEED

def b_vnd(inst: Instance,
          sol: Solution,
          neighborhoods=['N1','N2','N3','N4'],
          eval_f=None,
          max_total_iters=10000,
          sample_limit: int = None,
          debug_visual: bool = False) -> Tuple[Solution, bool, float, dict]:
    """
    deterministic B-VND: enumerates moves in each neighborhood
    on first improvement, accept and restart from neighborhood 1
    """

    if eval_f is None:
        eval_f = lambda s: evaluate_solution(inst, s)['F']

    if sample_limit is None:
        sample_limit = max(2000, len(inst.visits) * 20)
        
    current = clone_solution(sol)
    current_eval = evaluate_solution(inst, current)
    current_val = eval_f(current)
    total_iters = 0
    improved_flag = False

    if debug_visual:
        visualize_solution(
            inst,
            current,
            start_times=current_eval.get("start_times"),
            title="b_vnd_start",
            show=True,
        )

    # neighborhood enumerators
    def enum_N1(cur):
        for r_idx in range(len(cur.routes)):
            L = len(cur.routes[r_idx])
            if L < 1: continue
            for i in range(L):
                for j in range(L):
                    if i == j: continue
                    yield intra_route_insertion(cur, r_idx, i, j)
    def enum_N2(cur):
        for r1 in range(len(cur.routes)):
            for r2 in range(len(cur.routes)):
                if r1 == r2: continue
                if len(cur.routes[r1]) == 0: continue
                for p in range(len(cur.routes[r1])):
                    for inspos in range(len(cur.routes[r2]) + 1):
                        yield inter_route_insertion(cur, r1, p, r2, inspos)
    def enum_N3(cur):
        for r in range(len(cur.routes)):
            L = len(cur.routes[r])
            if L < 2: continue
            for i in range(L):
                for j in range(i+1, L):
                    yield intra_route_swap(cur, r, i, j)
    def enum_N4(cur):
        for r1 in range(len(cur.routes)):
            for r2 in range(r1+1, len(cur.routes)):
                if len(cur.routes[r1])==0 or len(cur.routes[r2])==0: continue
                for p1 in range(len(cur.routes[r1])):
                    for p2 in range(len(cur.routes[r2])):
                        yield inter_route_swap(cur, r1, p1, r2, p2)

    enum_map = {'N1': enum_N1, 'N2': enum_N2, 'N3': enum_N3, 'N4': enum_N4}

    #  main loop 
    while total_iters < max_total_iters:
        improved = False
        for nb in neighborhoods:
            gen = enum_map[nb](current)
            for sampled, cand in enumerate(gen, start=1):
                if sampled > sample_limit or total_iters >= max_total_iters:
                    break
                total_iters += 1

                cand_eval = evaluate_solution(inst, cand)
                cand_val = cand_eval['F']

                if cand_val + 1e-9 < current_val:
                    current, current_val, current_eval = cand, cand_val, cand_eval
                    improved = improved_flag = True

                    if debug_visual:
                        visualize_solution(
                            inst,
                            current,
                            start_times=current_eval.get("start_times"),
                            title=f"b_vnd_improve_{nb}",
                            show=True,
                        )
                    break  # restart from first neighborhood
            if improved:
                break
        if not improved:
            break

    # return eval dict so caller doesn’t need to recompute
    return current, improved_flag, current_val, current_eval

def update_penalties(w1, w2, PenT, PenH, alpha=0.03, w1_min=1e-6, w2_min=1e-6,
                     w1_max=None, w2_max=None, iter_no=0, update_every=1):
    """
    multiplicative update like the paper but:
      - alpha smaller by default (0.03),
      - optional caps (w1_max, w2_max),
      - optional update_every: only update weights every k iterations.
    it returns the updated (w1,w2).
    """
    if update_every > 1:
        if (iter_no % update_every) != 0:
            return w1, w2
    if PenT > 0:
        w1 = w1 * (1.0 + alpha)
    else:
        w1 = max(w1 * (1.0 - alpha), w1_min)

    if PenH > 0:
        w2 = w2 * (1.0 + alpha)
    else:
        w2 = max(w2 * (1.0 - alpha), w2_min)

    if w1_max is not None:
        w1 = min(w1, w1_max)
    if w2_max is not None:
        w2 = min(w2, w2_max)

    return w1, w2

def gvns(inst: Instance,
         Tmax_seconds=60,
         contmax=100,
         p_init=1,
         p_max=10,
         Dp=0.3,
         w1_init=0.5,
         w2_init=0.1,
         alpha=0.1,
         delta=0.4,
         M=1e6,
         w1_max=None, w2_max=None,
         update_every=1,
         seed=SEED,
         verbose=True):
    random.seed(seed)
    if M is None:
        max_tt = max(inst.travel_time.values()) if inst.travel_time else 1.0
        M = max_tt * max(1, len(inst.visits)) * 100.0

    # initial solution 
    current = random_insertion(inst, num_vehicles_estimate=5, seed=seed)
    eval_current = evaluate_solution(inst, current, w1_init, w2_init, M)
    feasibility = is_feasible(inst, current, eval_current)
    if not feasibility:
        if verbose:
            print("[gvns] initial solution infeasible → quick_repair_greedy")
        current, eval_current = quick_repair_greedy(inst, current, max_tries=500, verbose=verbose)
        feasibility = is_feasible(inst, current, eval_current)
        if verbose:
            print("[gvns] after quick_repair: f=%.2f PenT=%.2f F=%.2f"
                  % (eval_current['f'], eval_current['PenT'], eval_current['F']))
    else:
        if verbose:
            print("[gvns] initial solution feasible. f=%.2f" % eval_current['f'])

    # best overall (may be infeasible)
    best = clone_solution(current)
    eval_best = eval_current
    F_best = eval_best['F']

    #  track best feasible one   
    best_feasible = clone_solution(current) if feasibility else None
    f_best_feasible = eval_current['f'] if feasibility else float("inf")

    # initialize parameters
    w1, w2 = w1_init, w2_init
    p, cont, iter_no = p_init, 0, 0
    history = []
    t0 = time.time()

    while True:
        iter_no += 1
        if time.time() - t0 > Tmax_seconds: break
        if cont >= contmax:
            if verbose: print("Terminating: cont >= contmax")
            break

        # 1) shake
        shaken, _ = shake_solution(inst, current, p=p, Dp=Dp,
                                   seed=random.randint(0, 1_000_000))

        # 2) local search
        local, _, F_local, _ = b_vnd(
            inst, shaken,
            neighborhoods=['N1','N2','N3','N4'],
            eval_f=lambda s: evaluate_solution(inst, s, w1, w2, M)['F'],
            max_total_iters=20000
        )

        # 3) evaluate and update penalties
        eval_local = evaluate_solution(inst, local, w1, w2, M)
        w1, w2 = update_penalties(
            w1, w2, eval_local['PenT'], eval_local['PenH'],
            alpha=alpha, w1_max=w1_max, w2_max=w2_max,
            iter_no=iter_no, update_every=update_every
        )

        eval_local_after = evaluate_solution(inst, local, w1, w2, M)
        F_local_after = eval_local_after['F']

        eval_current = evaluate_solution(inst, current, w1, w2, M)
        F_current = eval_current['F']

        # 4) acceptance rule
        accepted = False
        if F_local_after <= (1.0 + delta) * F_current:
            current = local
            accepted = True
            cont = 0
            p = p_init

            #  update overall best 
            if eval_local_after['F'] < F_best:
                best = clone_solution(local)
                eval_best = eval_local_after
                F_best = eval_local_after['F']
                if verbose:
                    print(f"[{iter_no}] New best F={F_best:.3f} at t={time.time()-t0:.2f}s")

            #  update best feasible 
            if is_feasible(inst, local, eval_local_after):
                if eval_local_after['f'] < f_best_feasible:
                    best_feasible = clone_solution(local)
                    f_best_feasible = eval_local_after['f']
        else:
            cont += 1
            p = min(p + 1, p_max)

        history.append({
            'iter': iter_no, 'time': time.time() - t0,
            'F_current': F_current, 'F_local_after': F_local_after, 'F_best': F_best,
            'w1': w1, 'w2': w2,
            'PenT': eval_local_after['PenT'], 'PenH': eval_local_after['PenH'],
            'PenS': eval_local_after['PenS'],
            'sync_mismatch': eval_local_after.get('sync_time_diff', 0.0),
            'accepted': accepted
        })

    # final selection 
    if best_feasible is not None:
        if verbose:
            print(f"[gvns] Returning best feasible with f={f_best_feasible:.2f}")
        return best_feasible, history
    else:
        if verbose:
            print("[gvns] No feasible solution found, returning best penalised.")
        return best, history
