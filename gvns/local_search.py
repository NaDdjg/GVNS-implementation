from typing import Tuple
from .solution import Solution, set_deterministic
from .instance import Instance, paper_initial_fixed, SEED
from .operators import (intra_route_insertion, inter_route_insertion, two_opt, or_opt, cross_exchange,
                        intra_route_swap, inter_route_swap, clone_solution, shake_solution)
from .evaluate import evaluate_solution, quick_repair_greedy, is_feasible, repair_missing_and_sync
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
          max_total_iters= None,
          sample_limit: int = None,
          verbose: bool = False,
          debug_visual: bool = False) -> Tuple[Solution, bool, float, dict]:

    import math, heapq, random
    if eval_f is None:
        eval_f = lambda s: evaluate_solution(inst, s, w1, w2, M)['F']

    n_vis = max(1, len(getattr(inst, "visits", {})))

    # defaults (scale with instance size)
    if sample_limit is None:
        sample_limit = max(8000, int(len(inst.visits)**2 / 6))
    if max_total_iters is None:
        max_total_iters = sample_limit * 2

    sample_limit = int(sample_limit)
    max_total_iters = int(max_total_iters)

    current = clone_solution(sol)
    current_eval = evaluate_solution(inst, current, w1, w2, M)
    current_val = eval_f(current)
    total_iters = 0
    improved_flag = False

    if debug_visual:
        plot_side_by_side(inst, current, current, title="b_vnd_start",
                          move_desc="B-VND start", show=True)

    # --- helper heuristics -------------------------------------------------
    def route_center(route):
        # approximate centroid (x,y) of route endpoints -> cheap spatial heuristic
        if not route:
            return None
        a = inst.visits[route[0]]
        b = inst.visits[route[-1]]
        return ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)

    def euclid(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    # precompute route centers for heuristics (refreshed before neighborhood scans)
    def compute_centers(sol):
        return [route_center(r) for r in sol.routes]

    # sample a limited set of index pairs from two sequences (deterministic + random)
    def sample_index_pairs(len_a, len_b, max_pairs):
        pairs = []
        # always include small deterministic samples (start/end/median)
        cand_a = list({0, max(0, len_a-1), len_a//2} & set(range(len_a)))
        cand_b = list({0, max(0, len_b-1), len_b//2} & set(range(len_b)))
        for i in cand_a:
            for j in cand_b:
                pairs.append((i, j))
        # add some random pairs
        needed = max_pairs - len(pairs)
        for _ in range(max(0, needed)):
            pairs.append((random.randrange(len_a), random.randrange(len_b)))
        return pairs[:max_pairs]

    # --- neighborhood enumerators (heuristic-limited) ----------------------
    def enum_N1(cur, centers):
        # intra-route insertion + some 2-opt/or-opt local improvements
        for r_idx in range(len(cur.routes)):
            L = len(cur.routes[r_idx])
            if L < 1:
                continue
            # prioritize moves near route ends and median positions
            important_pos = list({0, max(0, L-1), L//2} & set(range(L)))
            # produce deterministic insertion candidates first
            for i in important_pos:
                for j in range(L):
                    if i == j: continue
                    yield intra_route_insertion(cur, r_idx, i, j)
            # then limited sample of other positions
            max_pairs = max(6, min(30, L*2))
            sampled_pairs = sample_index_pairs(L, L, max_pairs)
            for i,j in sampled_pairs:
                if i == j: continue
                yield intra_route_insertion(cur, r_idx, i, j)
            # small 2-opt locally (limit)
            if L >= 4:
                for i in range(0, min(4, L-2)):
                    for k in range(i+1, min(L-1, i+4)):
                        yield two_opt(cur, r_idx, i, k)
            # short or-opt block moves
            if L >= 2:
                for block in (1,):
                    for i in range(0, max(1, min(4, L - block + 1))):
                        for inspos in range(0, max(1, min(4, L - block + 1))):
                            if inspos == i: continue
                            yield or_opt(cur, r_idx, i, block, r_idx, inspos)

    def enum_N2(cur, centers):
        # inter-route insertion with spatial prefiltering and sampled pairs
        R = len(cur.routes)
        if R < 2: return
        # compute pair-wise center distances to favor near routes
        order = []
        for r1 in range(R):
            for r2 in range(R):
                if r1 == r2: continue
                c1, c2 = centers[r1], centers[r2]
                dist = euclid(c1, c2) if c1 and c2 else 0.0
                order.append((dist, r1, r2))
        # sort by increasing distance (prefer moving between close routes)
        order.sort()
        max_route_pairs = max(200, R * 4)
        pairs_considered = 0
        for _, r1, r2 in order:
            if pairs_considered >= max_route_pairs:
                break
            L1 = len(cur.routes[r1])
            L2 = len(cur.routes[r2])
            if L1 == 0: 
                continue
            # sample a limited number of positions (deterministic + random)
            max_pos_pairs = min(80, max(6, (L1 + L2)))
            idx_pairs = sample_index_pairs(L1, L2 + 1, max_pos_pairs)
            for p, inspos in idx_pairs:
                pairs_considered += 1
                yield inter_route_insertion(cur, r1, p, r2, inspos)
            # also try moving 2-length block (or_opt) from r1->r2
            for block in (1, 2):
                if L1 < block: continue
                for i in range(0, min(6, L1 - block + 1)):
                    for inspos in range(0, min(6, L2 + 1)):
                        yield or_opt(cur, r1, i, block, r2, inspos)

    def enum_N3(cur, centers):
        # intra-route swap + intra 2-opt (heuristic-limited)
        for r in range(len(cur.routes)):
            L = len(cur.routes[r])
            if L < 2: continue
            # try adjacent swaps and some sampled distant swaps
            for i in range(max(1, min(6, L-1))):
                j = (i+1) % L
                yield intra_route_swap(cur, r, i, j)
            max_pairs = min(50, L*(L-1)//2)
            sampled_pairs = sample_index_pairs(L, L, max_pairs)
            seen = set()
            for i,j in sampled_pairs:
                if i==j: continue
                key = (min(i,j), max(i,j))
                if key in seen: continue
                seen.add(key)
                yield intra_route_swap(cur, r, i, j)
            # occasional two-opt
            if L >= 4:
                for i in range(0, min(4, L-2)):
                    for k in range(i+1, min(L-1, i+4)):
                        yield two_opt(cur, r, i, k)

    def enum_N4(cur, centers):
        # inter-route swap + cross-exchange (powerful) with sampling and center-based order
        R = len(cur.routes)
        if R < 2: return
        # create route-pair ordered by center proximity
        route_pairs = []
        for r1 in range(R):
            for r2 in range(r1+1, R):
                c1, c2 = centers[r1], centers[r2]
                d = euclid(c1,c2) if c1 and c2 else 0.0
                route_pairs.append((d, r1, r2))
        route_pairs.sort()
        pairs_considered = 0
        for _, r1, r2 in route_pairs:
            if pairs_considered > max(200, R*4):
                break
            L1, L2 = len(cur.routes[r1]), len(cur.routes[r2])
            if L1 == 0 or L2 == 0: continue
            # sampled point swaps
            max_points = min(100, L1 * L2)
            idx_pairs = sample_index_pairs(L1, L2, min(30, max_points))
            for p1, p2 in idx_pairs:
                pairs_considered += 1
                yield inter_route_swap(cur, r1, p1, r2, p2)
            # cross-exchange (swap tails) - limited sampling of cut points
            if L1 >= 2 and L2 >= 2:
                max_cross = min(12, (L1-1)*(L2-1))
                cnt = 0
                for p1 in range(0, min(6, L1-1)):
                    for p2 in range(0, min(6, L2-1)):
                        if cnt >= max_cross:
                            break
                        cnt += 1
                        yield cross_exchange(cur, r1, p1, r2, p2)

    enum_map = {'N1': enum_N1, 'N2': enum_N2, 'N3': enum_N3, 'N4': enum_N4}

    # if PenT persists, increase budgets
    if current_eval.get('PenT', 0.0) > 1e-9:
        sample_limit *= 2
        max_total_iters *= 2

    # best-improvement frequency (more often for inter-route neighborhoods)
    base_best_every = max(3, int(max(1, len(inst.visits)//60)))

    # track neighborhood success to adapt local ordering
    neigh_success = {nb: 1e-6 for nb in neighborhoods}
    neigh_attempts = {nb: 1e-6 for nb in neighborhoods}

    nb_call_counter = 0

    # main loop
    while total_iters < max_total_iters:
        improved = False

        # internal order: sort neighborhoods by recent success rate but keep global order indices stable
        ordered_neighs = sorted(neighborhoods, key=lambda n: - (neigh_success.get(n,0)/neigh_attempts.get(n,1)))
        # but keep external given order priority for fairness: move top if improvement recently
        # we'll still iterate ordered_neighs (prioritize successful ones)
        centers = compute_centers(current)

        for nb in ordered_neighs:
            nb_call_counter += 1
            gen = enum_map[nb](current, centers)

            # heuristically decide do_best: do best more often for inter-route (N2,N4)
            do_best = ((nb in ('N2','N4')) and (nb_call_counter % max(2, base_best_every//2) == 0)) or \
                      (nb_call_counter % base_best_every == 0)

            neigh_sampled = 0
            best_cand = None
            best_cand_val = float('inf')

            # iterate candidates (either until first improvement or to collect best)
            for cand in gen:
                neigh_sampled += 1
                if neigh_sampled > sample_limit or total_iters >= max_total_iters:
                    break
                total_iters += 1

                cand_eval = evaluate_solution(inst, cand, w1, w2, M)
                cand_val = cand_eval['F']

                # immediate acceptance (first-improvement) if not doing best scan
                if (not do_best) and (cand_val + 1e-9 < current_val):
                    before = clone_solution(current)
                    current, current_eval = cand, cand_eval
                    current_val = cand_val
                    improved = improved_flag = True
                    neigh_success[nb] += 1
                    neigh_attempts[nb] += 1
                    if verbose:
                        print(f"[B-VND] improvement found by {nb}: new F={current_val:.6g} (sampled={neigh_sampled})")
                    if debug_visual:
                        diff = which_nodes_changed(before, current)
                        highlight = set(diff['added'] + diff['removed'] + [d[0] for d in diff['moved']])
                        desc = f"B-VND {nb} -> F={current_val:.3f} | changed={len(highlight)}"
                        savepath = plot_side_by_side(inst, before, current,
                                                     title=f"b_vnd_improve_{nb}",
                                                     highlight_nodes=highlight,
                                                     move_desc=desc, show=True)
                        if verbose:
                            print("Saved:", savepath)
                    break

                # otherwise collect best candidate
                if cand_val + 1e-9 < best_cand_val:
                    best_cand_val = cand_val
                    best_cand = (cand, cand_eval)

            # end candidate loop
            neigh_attempts[nb] += max(1, neigh_sampled)

            if (not improved) and do_best and best_cand is not None and best_cand_val + 1e-9 < current_val:
                before = clone_solution(current)
                current, current_eval = best_cand
                current_val = best_cand_val
                improved = improved_flag = True
                neigh_success[nb] += 1
                if verbose:
                    print(f"[B-VND best] improvement found by {nb}: new F={current_val:.6g} (collected={neigh_sampled})")
                if debug_visual:
                    diff = which_nodes_changed(before, current)
                    highlight = set(diff['added'] + diff['removed'] + [d[0] for d in diff['moved']])
                    desc = f"B-VND best {nb} -> F={current_val:.3f} | changed={len(highlight)}"
                    savepath = plot_side_by_side(inst, before, current,
                                                 title=f"b_vnd_improve_best_{nb}",
                                                 highlight_nodes=highlight,
                                                 move_desc=desc, show=True)
                    if verbose:
                        print("Saved:", savepath)

            if improved:
                # after improvement restart scanning neighborhoods (standard VND behavior)
                break

        if not improved:
            # no improvement in any neighborhood -> end local search
            break

    # ensure we have current_eval consistent
    if 'current_eval' not in locals() or current_eval is None:
        current_eval = evaluate_solution(inst, current, w1, w2, M)
        current_val = eval_f(current)

    return current, improved_flag, current_val, current_eval


def update_penalties(
    w1, w2, PenT, PenH,
    alpha=0.1,
    w_min=1e-4,
    w_max=0.9,
    iter_no=0,
    update_every=1,
    normalize=True
):
    """
    Adaptive penalty update for GVNS:
    - Increases weight if corresponding penalty > 0,
    - Decreases if no violation,
    - Keeps (w1 + w2) ≈ 1 for stability.
    """
    if update_every > 1 and (iter_no % update_every) != 0:
        return w1, w2

    # proportional adjustment
    if PenT > 1e-6:
        w1 = min(w1 * (1 + alpha * (1 + PenT / (1 + PenH))), w_max)
    else:
        w1 = max(w1 * (1 - alpha * 0.5), w_min)

    if PenH > 1e-6:
        w2 = min(w2 * (1 + alpha * (1 + PenH / (1 + PenT))), w_max)
    else:
        w2 = max(w2 * (1 - alpha * 0.5), w_min)

    # --- normalize so w1 + w2 = 1 ---
    if normalize:
        total = w1 + w2
        if total > 0:
            w1 /= total
            w2 /= total

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
         num_vehicles_estimate=60,
         agent=None,
         verbose=False
        ):
    
    if M is None:
        try:
            max_tt = max(inst.travel_time.values()) if inst.travel_time else 1.0
        except Exception:
            max_tt = 1.0
        M = max_tt * max(1, len(inst.visits)) * 10.0

    set_deterministic(seed)    

    current, eval_current = paper_initial_fixed(inst, num_vehicles_estimate=num_vehicles_estimate, seed=seed, avoid_pair=True)
    # current, eval_current = quick_repair_greedy(inst, current, max_tries=1000, verbose=verbose)
    # if eval_current['PenS'] > 0:
    #     current, eval_current = repair_missing_and_sync(inst, current, verbose=True)

    print(f"[init] After repair_missing_and_sync: PenS={eval_current['PenS']}, sync_pairs={getattr(inst, 'sync_pairs', [])}")

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
    no_improve_iters = 0

    neigh_attempts = [0.0] * len(NEIGHBORHOODS)
    neigh_success = [0.0] * len(NEIGHBORHOODS)

    while True:
        iter_no += 1
        if Tmax_seconds is not None and time.time() - t0 > Tmax_seconds: 
            if verbose:
                print("[gvns] Terminating: Tmax reached")
            break
        if cont >= contmax:
            if verbose: print("Terminating: cont >= contmax")
            break

        # 1) shake
        strong_shake = False
        if eval_current.get('PenT', 0.0) > 1e-9 and stagn_penT_iters >= 5:
            strong_shake = True

        if no_improve_iters > 8:
            p = min(p * 2, p_max)  # double up to p_max
        if strong_shake:
            shake_p = max(p, min(p_max, int(p * 1.5)))
            shake_Dp = min(0.8, Dp + 0.15)
        else:
            shake_p = p
            shake_Dp = Dp

        shaken, _ = shake_solution(inst, current, p=shake_p, Dp=shake_Dp,
                                   seed=random.randint(0, 1_000_000), verbose=verbose, debug_visual=False)

        # 2) local search
        default_sample = max(10000, int(len(inst.visits)**2 / 4))
        b_max_iters = (default_sample * 2) * (3 if strong_shake or eval_current.get('PenT',0.0)>1e-9 else 1)
            
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
            sample_limit=default_sample,
            verbose=verbose,
            debug_visual=False
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
            cont = 0
            p = max(p_init, 1)
            accepted = True
            no_improve_iters = 0

            # update overall best 
            if eval_current['F'] < F_best:
                best = clone_solution(local)
                eval_best = eval_current
                F_best = eval_current['F']
                if verbose:
                    print(f"[{iter_no}] New best F={F_best:.3f} at t={time.time()-t0:.2f}s")

            # update best feasible if applicable
            if is_feasible(inst, local, eval_current) and eval_current['f'] < f_best_feasible:
                best_feasible = clone_solution(local)
                f_best_feasible = eval_current['f']

        # if not accepted, increment counters 
        if not accepted:
            cont += 1
            p = min(p + 1, p_max)
            no_improve_iters += 1

        # 5) update penalties
        w1, w2 = update_penalties(
            w1, w2, eval_current['PenT'], eval_current['PenH'],
            alpha=alpha, iter_no=iter_no, update_every=update_every, 
        )

        # check PenT stagnation
        curPenT = eval_current.get('PenT', 0.0)
        if abs(curPenT - last_penT) < 1e-6:
            stagn_penT_iters += 1
        else:
            stagn_penT_iters = 0
        last_penT = curPenT

        penS = eval_current.get('PenS', 0.0)
        penT = eval_current.get('PenT', 0.0)

        if iter_no > 8 and stagn_penT_iters >= 8 and eval_current['PenT'] > 1e-9:
            print("[gvns] Quick repair triggered: PenT stagnation detected (PenS=0).")
            try_repair_sol, try_repair_eval = quick_repair_greedy(inst, current, max_tries=5000, verbose=verbose)
            if try_repair_eval['F'] + 1e-9 < eval_current['F'] or \
            try_repair_eval['PenT'] + try_repair_eval['PenH'] < eval_current['PenT'] + eval_current['PenH']:
                current, eval_current = try_repair_sol, try_repair_eval
                stagn_penT_iters = 0

        elif iter_no > 8 and penS > 1e-9:
            print("[gvns] Full repair triggered: PenS>0 detected.")
            try_repair_sol, try_repair_eval = quick_repair_greedy(inst, current, max_tries=1000, verbose=verbose)
            fix_sol, fix_eval = repair_missing_and_sync(inst, try_repair_sol, verbose=verbose)
            if (fix_eval['F'] + 1e-9 < eval_current['F'] or
                (fix_eval['PenS'] + fix_eval['PenT'] + fix_eval['PenH']) <
                (eval_current['PenS'] + eval_current['PenT'] + eval_current['PenH'])):
                current, eval_current = fix_sol, fix_eval
                stagn_penT_iters = 0

        # 6) Re-evaluate current with updated weights for logging/history
        eval_current_after = evaluate_solution(inst, current, w1, w2, M)
        raw_eval = evaluate_solution(inst, current, 0.0, 0.0, 0.0)
        raw_f = raw_eval['f']

        # to make the search focus on travel minimization afetr feasibility
        if eval_current['PenS'] < 1e-9 and eval_current['PenT'] < 1e-9:
            M = max(M * 0.5, 1e4)

        history.append({
            'iter': iter_no, 'time': time.time() - t0,
            'F_current': eval_current_after['F'], 'F_local_after': F_local_after, 'F_best': F_best,
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
        
    #  final selection 
    if best_feasible is not None:
        if verbose:
            print(f"[gvns] Returning best feasible with f={f_best_feasible:.2f}")
        return best_feasible, history
    else:
        if verbose:
            print("[gvns] No feasible solution found, returning best penalised.")
        return best, history