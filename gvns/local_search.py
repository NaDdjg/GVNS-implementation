from typing import Tuple
from .solution import Solution
from .instance import Instance
from .operators import (intra_route_insertion, inter_route_insertion,
                        intra_route_swap, inter_route_swap, clone_solution, shake_solution)
from .evaluate import evaluate_solution, quick_repair_greedy, is_feasible
from .viz import plot_side_by_side, which_nodes_changed
import random, time
from .instance import random_insertion, SEED

def build_initial_solution(
    inst: Instance,
    num_vehicles_estimate: int = 5,
    seed: int = SEED,
    max_random_attempts: int = 10,
    repair_max_tries: int = 500,
    greedy_w1: float = 100.0,
    greedy_w2: float = 10.0,
    greedy_M: float = None,
    verbose: bool = False
):
    """
    Build an initial solution trying, in order:
      1) several random_insertion() starts followed by quick_repair_greedy()
      2) a constructive greedy insertion (minimize penalized F during insertion)
      3) final quick_repair_greedy() attempt on the greedy result

    Returns: (Solution, eval_dict) where eval_dict = evaluate_solution(inst, sol, ...)
    """

    # helpers
    def _violates_same_route(candidate, route_idx, routes):
        # mimic the same rule used in your random_insertion:
        for (a, b) in getattr(inst, "sync_pairs", []):
            if candidate == a and b in routes[route_idx]:
                return True
            if candidate == b and a in routes[route_idx]:
                return True
        return False

    # compute fallback large M for greedy if not provided
    if greedy_M is None:
        try:
            # inst.travel_time might be dict -> take max value
            max_tt = max(inst.travel_time.values()) if getattr(inst, "travel_time", None) else 1.0
            n_vis = max(1, len(getattr(inst, "visits", {})))
            greedy_M = max_tt * n_vis * 100.0
        except Exception:
            greedy_M = 1e6

    best_overall_sol = None
    best_overall_eval = None

    #  1) Try several random starts + repair 
    for attempt in range(max_random_attempts):
        s_rand = random_insertion(inst, num_vehicles_estimate=num_vehicles_estimate, seed=seed + attempt * 31)
        ev_rand = evaluate_solution(inst, s_rand, 0.5, 0.1, greedy_M)  # lightweight eval
        if is_feasible(inst, s_rand, ev_rand):
            if verbose:
                print(f"[init] Random start {attempt} already feasible.")
            return s_rand, ev_rand

        # attempt repair
        repaired, ev_repaired = quick_repair_greedy(inst, s_rand, max_tries=repair_max_tries, verbose=verbose)
        if is_feasible(inst, repaired, ev_repaired):
            if verbose:
                print(f"[init] Repaired feasible from random start {attempt}.")
            return repaired, ev_repaired

        # keep best penalised solution as fallback
        if best_overall_eval is None or ev_repaired["F"] < best_overall_eval["F"]:
            best_overall_sol, best_overall_eval = repaired, ev_repaired

    #  2) Greedy constructive insertion fallback 
    # deterministic ordering: by earliest time window
    customers = sorted(inst.visits.keys(), key=lambda i: inst.visits[i].tw_early)

    # start with estimated empty routes
    routes = [[] for _ in range(num_vehicles_estimate)]

    # build incrementally
    for vid in customers:
        best_local_val = float("inf")
        best_local_routes = None

        # try inserting into all existing routes at all positions
        for ridx in range(len(routes)):
            # skip route positions if violates same-route rule
            if _violates_same_route(vid, ridx, routes):
                # still try other positions in other routes; we skip this entire route
                continue
            for pos in range(len(routes[ridx]) + 1):
                cand_routes = [list(r) for r in routes]
                cand_routes[ridx].insert(pos, vid)
                cand = clone_solution(Solution(routes=cand_routes))
                ev_c = evaluate_solution(inst, cand, greedy_w1, greedy_w2, greedy_M)
                if ev_c["F"] < best_local_val:
                    best_local_val = ev_c["F"]
                    best_local_routes = cand_routes

        # try creating new route for this vid
        cand_routes = [list(r) for r in routes]
        cand_routes.append([vid])
        cand = clone_solution(Solution(routes=cand_routes))
        ev_c = evaluate_solution(inst, cand, greedy_w1, greedy_w2, greedy_M)
        if ev_c["F"] < best_local_val:
            best_local_val = ev_c["F"]
            best_local_routes = cand_routes

        # if we found any insertion, adopt it
        if best_local_routes is not None:
            routes = best_local_routes
        else:
            # As a safety: append to last route if nothing else (unlikely)
            routes[-1].append(vid)

    greedy_sol = clone_solution(Solution(routes=[list(r) for r in routes]))
    greedy_eval = evaluate_solution(inst, greedy_sol, greedy_w1, greedy_w2, greedy_M)
    if verbose:
        print("[init] Greedy built solution: F=%.3f f=%.3f PenT=%.3f" % (greedy_eval["F"], greedy_eval["f"], greedy_eval["PenT"]))

    #  3) Final repair attempt on greedy solution 
    repaired2, ev_repaired2 = quick_repair_greedy(inst, greedy_sol, max_tries=repair_max_tries, verbose=verbose)
    if is_feasible(inst, repaired2, ev_repaired2):
        if verbose:
            print("[init] Greedy+repair produced feasible solution.")
        return repaired2, ev_repaired2

    # if still not feasible, return best among all attempts (prefer feasible if any)
    candidates = [(best_overall_sol, best_overall_eval), (greedy_sol, greedy_eval), (repaired2, ev_repaired2)]
    # filter out None
    candidates = [(s, e) for (s, e) in candidates if s is not None and e is not None]
    # find min F
    best = min(candidates, key=lambda se: se[1]["F"]) if candidates else (greedy_sol, greedy_eval)
    if verbose:
        print("[init] Returning best penalised initial solution (F=%.3f)." % best[1]["F"])
    return best


def b_vnd(inst: Instance,
          sol: Solution,
          neighborhoods=['N1','N2','N3','N4'],
          eval_f=None,
          w1: float = 0.5,
          w2: float = 0.1, 
          M: float = None,
          max_total_iters=10000,
          sample_limit: int = None,
          verbose: bool = False,
          debug_visual: bool = False) -> Tuple[Solution, bool, float, dict]:
    """
    deterministic B-VND: enumerates moves in each neighborhood
    on first improvement, accept and restart from neighborhood 1
    """

    if eval_f is None:
        eval_f = lambda s: evaluate_solution(inst, s)['F']

    n_vis = max(1, len(getattr(inst, "visits", {})))
    if sample_limit is None:
        sample_limit = max(2000, len(inst.visits) * 25)
    
    current = clone_solution(sol)
    current_eval = evaluate_solution(inst, current, w1, w2, M)
    current_val = eval_f(current)
    total_iters = 0
    improved_flag = False

    if debug_visual:
        plot_side_by_side(inst, current, current, title="b_vnd_start",
                          move_desc="B-VND start", show=True)

    # focused repair phase if PenT > 0
    # if current_eval.get('PenT', 0.0) > 1e-9:
    #     # collect offending visits
    #     offending = set()
    #     for r_sched in current_eval.get('route_schedules', []):
    #         for (vid, _, start, _) in r_sched:
    #             if start > inst.visits[vid].tw_late + 1e-9:
    #                 offending.add(vid)
    #     # also include missing visits (PenS) - though PenS usually zero here
    #     # limit attempts
    #     focused_tries = min(2000, max(200, n_vis * 5))
    #     tries = 0
    #     if verbose:
    #         print(f"[B-VND] focused repair phase: offending count={len(offending)} tries={focused_tries}")
    #     # enumerate some targeted moves involving offending nodes
    #     while tries < focused_tries and offending:
    #         tries += 1
    #         # pick an offending visit and try a few insertion/relocation positions
    #         vid = random.choice(list(offending))
    #         # find route & pos
    #         src_r = None; src_pos = None
    #         for ri, r in enumerate(current.routes):
    #             if vid in r:
    #                 src_r = ri
    #                 src_pos = r.index(vid)
    #                 break

    #         # try moving vid to other routes and earlier positions within same route
    #         candidates = []

    #         if src_r is None:
    #             continue

    #         # try inserting earlier in same route
    #         for to_pos in range(max(0, src_pos - 5), src_pos):
    #             candidates.append(intra_route_insertion(current, src_r, src_pos, to_pos))
            
    #         # move to other routes
    #         for tr in range(len(current.routes)):
    #             if tr == src_r: 
    #                 continue
    #             for inspos in range(len(current.routes[tr]) + 1):
    #                 candidates.append(inter_route_insertion(current, src_r, src_pos, tr, inspos))
    #         # try or-opt small chunk around vid
    #         L = len(current.routes[src_r])
    #         if L > 1:
    #             start = max(0, src_pos - 1)
    #             end = min(L, start + 2)
    #             for tr in range(len(current.routes)):
    #                 for inspos in range(len(current.routes[tr]) + 1):
    #                     candidates.append(relocate_subroute(current, src_r, start, end, tr, inspos))

    #         # accept first candidate with lower PenT or F
    #         for cand in candidates:
    #             total_iters += 1
    #             cand_eval = evaluate_solution(inst, cand, w1, w2, M)
    #             if (cand_eval['PenT'] + 1e-9 < current_eval['PenT'] or
    #                     cand_eval['F'] + 1e-9 < current_val):
    #                 current, current_eval, current_val = cand, cand_eval, eval_f(cand)
    #                 improved_flag = True
    #                 if verbose:
    #                     print(f"[B-VND focused] accepted -> PenT {cand_eval['PenT']:.3f} F {current_val:.3f}")
    #                 break

    #         # refresh offending set
    #         offending = {
    #             v for r_sched in current_eval.get('route_schedules', [])
    #             for (v, a, s, d) in r_sched
    #             if s > inst.visits[v].tw_late + 1e-9
    #         }

    #         if total_iters >= max_total_iters:
    #             break

    # neighborhood enumerators (same as your code)
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

    # if PenT persists, give more sample budget
    if current_eval.get('PenT', 0.0) > 1e-9:
        sample_limit *= 2
        max_total_iters *= 2

    # main loop
    while total_iters < max_total_iters:
        improved = False
        for nb in neighborhoods:
            gen = enum_map[nb](current)
            for sampled, cand in enumerate(gen, start=1):
                if sampled > sample_limit or total_iters >= max_total_iters:
                    break
                total_iters += 1

                cand_eval = evaluate_solution(inst, cand, w1, w2, M)
                cand_val = cand_eval['F']

                if cand_val + 1e-9 < current_val:
                    # accepted improvement
                    before = clone_solution(current)
                    current, current_val, current_eval = cand, cand_val, cand_eval
                    improved = improved_flag = True

                    # print & visualize
                    if verbose:
                        print(f"[B-VND] improvement found by {nb}: new F={current_val:.6g} (sampled={sampled})")
                    if debug_visual:
                        diff = which_nodes_changed(before, current)
                        highlight = set(diff['added'] + diff['removed'] + [d[0] for d in diff['moved']])
                        desc = f"B-VND {nb} -> F={current_val:.3f} | changed={len(highlight)}"
                        savepath = plot_side_by_side(inst, before, current,
                                                     title=f"b_vnd_improve_{nb}",
                                                     highlight_nodes=highlight,
                                                     move_desc=desc, show=True)
                        print("Saved:", savepath)

                    # restart from first neighborhood
                    break
            if improved:
                break
        if not improved:
            break

    return current, improved_flag, current_val, current_eval

def update_penalties(w1, w2, PenT, PenH, alpha=0.1, w1_min=1e-6, w2_min=1e-6,
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
         Tmax_seconds=None,
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
        num_vehicles_estimate=30,
         verbose=False):
    random.seed(seed)
    if M is None:
        max_tt = max(inst.travel_time.values()) if inst.travel_time else 1.0
        M = max_tt * max(1, len(inst.visits)) * 100.0

    current = random_insertion(inst, num_vehicles_estimate=num_vehicles_estimate, seed=seed)
    eval_current = evaluate_solution(inst, current, w1_init, w2_init, M)
    feasibility = is_feasible(inst, current, eval_current)
    if not feasibility:
        if verbose:
            print("[gvns] initial solution infeasible → quick_repair_greedy")
        current, eval_current = quick_repair_greedy(inst, current, max_tries=1000, verbose=verbose)
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

    # adaptive counters
    stagn_penT_iters = 0
    last_penT = eval_current.get('PenT', 0.0)

    while True:
        iter_no += 1
        if Tmax_seconds is not None and time.time() - t0 > Tmax_seconds: break
        if cont >= contmax:
            if verbose: print("Terminating: cont >= contmax")
            break

        # 1) shake
        strong_shake = False
        if eval_current.get('PenT', 0.0) > 1e-9:
            if stagn_penT_iters >= 5:
                strong_shake = True

        shaken, _ = shake_solution(inst, current, p=p, Dp=Dp,
                                   seed=random.randint(0, 1_000_000), verbose=verbose, debug_visual=verbose)

        # 2) local search
        b_max_iters = 20000 * (3 if strong_shake or eval_current.get('PenT',0.0)>1e-9 else 1)
        local, _, F_local, _ = b_vnd(
            inst, shaken,
            neighborhoods=['N1','N2','N3','N4'],
            eval_f=lambda s: evaluate_solution(inst, s, w1, w2, M)['F'],
            w1=w1, w2=w2, M=M,
            max_total_iters=b_max_iters,
            verbose=verbose,
            debug_visual=verbose
        )

        # 3) evaluate
        eval_local = evaluate_solution(inst, local, w1, w2, M)
        F_local_after = eval_local['F']

        eval_current = evaluate_solution(inst, current, w1, w2, M)
        F_current = eval_current['F']

        # 4) acceptance rule
        accepted = False
        if F_local_after <= (1.0 + delta) * F_current:
            current = local
            eval_current = eval_local
            accepted = True
            cont = 0
            p = p_init

            #  update overall best 
            if eval_current['F'] < F_best:
                best = clone_solution(local)
                eval_best = eval_current
                F_best = eval_current['F']
                if verbose:
                    print(f"[{iter_no}] New best F={F_best:.3f} at t={time.time()-t0:.2f}s")
                    print(f"[{iter_no}] New best f={eval_best:.3f} at t={time.time()-t0:.2f}s")

            #  update best feasible 
            if is_feasible(inst, local, eval_current):
                if eval_current['f'] < f_best_feasible:
                    best_feasible = clone_solution(local)
                    f_best_feasible = eval_current['f']
        else:
            cont += 1
            p = min(p + 1, p_max)

        # 5) update penalties
        w1, w2 = update_penalties(
            w1, w2, eval_current['PenT'], eval_current['PenH'],
            alpha=alpha, w1_max=w1_max, w2_max=w2_max,
            iter_no=iter_no, update_every=update_every
        )

        # check PenT stagnation
        curPenT = eval_current.get('PenT', 0.0)
        if abs(curPenT - last_penT) < 1e-9:
            stagn_penT_iters += 1
        else:
            stagn_penT_iters = 0
        last_penT = curPenT

        # if PenT persists and stagnated, try a quick_repair_greedy attempt to aggressively remove offending nodes
        if stagn_penT_iters >= 8 and curPenT > 1e-9:
            if verbose:
                print("[gvns] PenT stagnation detected -> trying quick_repair_greedy aggressive pass")
            try_repair_sol, try_repair_eval = quick_repair_greedy(inst, current, max_tries=1000, verbose=verbose)
            if try_repair_eval['PenT'] + 1e-9 < curPenT or try_repair_eval['F'] + 1e-9 < F_current:
                current, eval_current = try_repair_sol, try_repair_eval
                if verbose:
                    print("[gvns] quick_repair improved PenT/F - accepted")

        # 6) Re-evaluate current with updated weights for logging/history
        eval_current_after = evaluate_solution(inst, current, w1, w2, M)
        raw_eval = evaluate_solution(inst, current, 0.0, 0.0, 0.0)
        raw_f = raw_eval['f']

        history.append({
            'iter': iter_no, 'time': time.time() - t0,
            'F_current': eval_current_after['F'], 'F_local_after': F_local, 'F_best': F_best,
            'f_best': eval_best['f'],
            'raw_f': raw_f,
            'w1': w1, 'w2': w2,
            'PenT': eval_current_after['PenT'], 'PenH': eval_current_after['PenH'],
            'PenS': eval_current_after['PenS'],
            'sync_mismatch': eval_current_after.get('sync_time_diff', 0.0),
            'accepted': accepted
        })

        print(f"[iter {iter_no:5d}] time={time.time()-t0:7.1f}s  "
                  f"F_current={eval_current_after['F']:8.2f}"
                  f"  best_feasible_f={f_best_feasible if best_feasible is not None else float('inf'):8.2f}"
                  f"  raw_f={raw_f:8.2f}  "
                  f"  w1={w1:.4f} w2={w2:.4f}  p={p}  cont={cont}"
                  f"  PenT={eval_current_after['PenT']:.2f} PenH={eval_current_after['PenH']:.2f} PenS={eval_current_after['PenS']:.2f}")
        
    # final selection 
    if best_feasible is not None:
        if verbose:
            print(f"[gvns] Returning best feasible with f={f_best_feasible:.2f}")
        return best_feasible, history
    else:
        if verbose:
            print("[gvns] No feasible solution found, returning best penalised.")
        return best, history
