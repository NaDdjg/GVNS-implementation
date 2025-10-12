from typing import Tuple
from .solution import Solution
from .instance import Instance, paper_initial_fixed, SEED
from .operators import (intra_route_insertion, inter_route_insertion,
                        intra_route_swap, inter_route_swap, clone_solution, shake_solution)
from .evaluate import evaluate_solution, quick_repair_greedy, is_feasible
from .viz import plot_side_by_side, which_nodes_changed
from .utils import NEIGHBORHOODS, extract_state
import random, time

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
         agent=None,
         verbose=False
        ):
    random.seed(seed)
    if M is None:
        max_tt = max(inst.travel_time.values()) if inst.travel_time else 1.0
        M = max_tt * max(1, len(inst.visits)) * 100.0

    # current = random_insertion(inst, num_vehicles_estimate=num_vehicles_estimate, seed=seed)
    # eval_current = evaluate_solution(inst, current, w1_init, w2_init, M)

    current, eval_current = paper_initial_fixed(inst, num_vehicles_estimate=num_vehicles_estimate, seed=seed, avoid_pair=True)
    current, eval_current = quick_repair_greedy(inst, current, max_tries=1000, verbose=verbose)

    feasibility = is_feasible(inst, current, eval_current)
    
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

    neigh_attempts = [0.0] * len(NEIGHBORHOODS)
    neigh_success = [0.0] * len(NEIGHBORHOODS)

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

        # rl integration
        if agent is not None:
            # Build state vector like in BVNDEnv
            state = extract_state(inst, current, eval_current, iter_no, t0, neigh_success, neigh_attempts)
            action = agent.select_action(state)
            chosen_neigh = [NEIGHBORHOODS[action]]
            action_id = action
        else:
            chosen_neigh = ['N1', 'N2', 'N3', 'N4']
            action_id = -1

        # bvnd call
        local, _, F_local, _ = b_vnd(
            inst, shaken,
            neighborhoods=chosen_neigh,
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

        # RL online update
        if agent is not None:
            # Compute reward based on improvement
            F_before = float(eval_current['F'])
            F_after = float(eval_local['F'])
            improved = F_after + 1e-9 < F_before
            reward = (F_before - F_after) / (1.0 + abs(F_before))
            if not improved:
                reward -= 0.01  # small penalty for failed improvement

            # reward -= 0.001 * (time.time() - t0)  

            # extract the next state
            next_state = extract_state(inst, local, eval_local, iter_no, start_time=t0)

            # update online agent
            try:
                agent.update(state, action, reward, next_state, done=False)
            except Exception as e:
                print("[RL-Update] warning:", e)

            neigh_attempts[action] += 1
            if improved:
                neigh_success[action] += 1

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
            'chosen_action': action_id,
            'chosen_neigh': chosen_neigh[0] if agent is not None else "classic",
            'w1': w1, 
            'w2': w2,
            'PenT': eval_current_after['PenT'], 
            'PenH': eval_current_after['PenH'],
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
