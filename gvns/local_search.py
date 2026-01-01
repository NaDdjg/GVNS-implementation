import random
import time
from typing import Optional

import gvns.operators as ops
from gvns.solution import Solution

# ============================================================
#  Lexicographic comparator 
# ============================================================
def is_better(new: Solution, old: Solution, eps_f=1e-9) -> bool:
    """
    lexicographic comparison:
      1) penS
      2) penT
      3) penH
      4) F travel time
    worsening enalties is done by shaking, and controlled by acceptance and penalty adaptation
    the role of bvnd is to find the best feasible solution possible
    """
    # no violation in synch allowed
    if new.penS < old.penS - 1e-12: return True
    if new.penS > old.penS + 1e-12: return False

    # among solutions with same penS, prefer lower penT
    if new.penT < old.penT - 1e-12: return True
    if new.penT > old.penT + 1e-12: return False

    # among solutions with same penS and penT, prefer lower penH
    if new.penH < old.penH - 1e-12: return True
    if new.penH > old.penH + 1e-12: return False

    # finally, among solutions with same penS, penT, penH, prefer lower Fval
    return new.Fval < old.Fval - eps_f

# ============================================================
# neighborhood explorers (best or first improvement)
# ============================================================
def explore_intra_insertion(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N1 - intra-route insertion."""
    m = sol.instance.m

    checks = 0
    improved = None

    for r in range(m):
        route = sol.tours[r]
        L = len(route)
        if L < 2: continue

        for i in range(L):
            for j in range(L + 1):
                if i == j: continue
                checks += 1
                if max_checks and checks > max_checks: return improved

                cand = ops.intra_insertion(sol, r, i, j)
                if not cand: continue
                cand.eval()

                if first_improve:
                    if is_better(cand, sol):
                        return cand
                else:
                    if improved is None or is_better(cand, improved):
                        improved = cand

    return improved


def explore_inter_insertion(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N2 - inter-route insertion."""
    m = sol.instance.m
    checks = 0
    improved = None

    for r1 in range(m):
        L1 = len(sol.tours[r1])
        if L1 == 0: continue

        for r2 in range(m):
            if r1 == r2: continue

            L2 = len(sol.tours[r2])
            for pos1 in range(L1):
                for pos2 in range(L2+1):

                    checks += 1
                    if max_checks and checks > max_checks: return improved

                    cand = ops.inter_insertion(sol, r1, pos1, r2, pos2)
                    if not cand: continue
                    cand.eval()

                    if first_improve:
                        if is_better(cand, sol):
                            return cand
                    else:
                        if improved is None or is_better(cand, improved):
                            improved = cand


    return improved

def explore_intra_swap(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N3 - intra-route swap."""
    m = sol.instance.m
    improved = None
    checks = 0

    for r in range(m):
        L = len(sol.tours[r])
        if L < 2: continue

        for i in range(L):
            for j in range(i+1, L):
                checks += 1
                if max_checks and checks > max_checks: return improved

                cand = ops.intra_swap(sol, r, i, j)
                if not cand: continue
                cand.eval()

                if first_improve:
                    if is_better(cand, sol):
                        return cand
                else:
                    if improved is None or is_better(cand, improved):
                        improved = cand

    return improved


def explore_inter_swap(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N4 - inter-route swap."""
    m = sol.instance.m
    improved = None
    checks = 0

    for r1 in range(m):
        for r2 in range(r1+1, m):
            L1 = len(sol.tours[r1])
            L2 = len(sol.tours[r2])
            if L1 == 0 or L2 == 0: continue

            for i in range(L1):
                for j in range(L2):
                    checks += 1
                    if max_checks and checks > max_checks: return improved

                    cand = ops.inter_swap(sol, r1, i, r2, j)
                    if not cand: continue
                    cand.eval()
                    
                    if first_improve:
                        if is_better(cand, sol):
                            return cand
                    else:
                        if improved is None or is_better(cand, improved):
                            improved = cand

    return improved


def explore_two_opt(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N5 - intra-route 2-opt"""
    m = sol.instance.m
    improved = None
    checks = 0

    for r in range(m):
        L = len(sol.tours[r])
        if L < 4: continue

        for i in range(L-2):
            for j in range(i+2, L):
                checks += 1
                if max_checks and checks > max_checks: return improved

                cand = ops.two_opt(sol, r, i, j)
                if not cand: continue
                cand.eval()

                if first_improve:
                    if is_better(cand, sol):
                        return cand
                else:
                    if improved is None or is_better(cand, improved):
                        improved = cand

    return improved


def explore_two_opt_star(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N6 - two-opt*"""
    m = sol.instance.m
    improved = None
    checks = 0

    for r1 in range(m):
        for r2 in range(r1+1, m):
            L1 = len(sol.tours[r1])
            L2 = len(sol.tours[r2])
            if L1 < 2 or L2 < 2: continue

            for i in range(1, L1):
                for j in range(1, L2):
                    checks += 1
                    if max_checks and checks > max_checks: return improved

                    cand = ops.two_opt_star(sol, r1, i, r2, j)
                    if not cand: continue
                    cand.eval()

                    if first_improve:
                        if is_better(cand, sol):
                            return cand
                    else:
                        if improved is None or is_better(cand, improved):
                            improved = cand


    return improved


def explore_cross_exchange(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N7 - cross exchange"""
    m = sol.instance.m
    improved = None
    checks = 0

    for r1 in range(m):
        for r2 in range(r1+1, m):
            L1 = len(sol.tours[r1])
            L2 = len(sol.tours[r2])
            if L1 < 2 or L2 < 2: continue

            for a1 in range(L1):
                for b1 in range(a1+1, min(a1+4, L1)+1):  # limit segment size
                    for a2 in range(L2):
                        for b2 in range(a2+1, min(a2+4, L2)+1):

                            checks += 1
                            if max_checks and checks > max_checks: return improved

                            cand = ops.cross_exchange(sol, r1, a1, b1, r2, a2, b2)
                            if not cand: continue
                            cand.eval()

                            if first_improve:
                                if is_better(cand, sol):
                                    return cand
                            else:
                                if improved is None or is_better(cand, improved):
                                    improved = cand
    return improved

def explore_k_relocate(sol: Solution, max_checks=None, first_improve: bool = False) -> Optional[Solution]:
    """N8 - k-relocate (move small segments between routes)"""
    m = sol.instance.m
    improved = None
    checks = 0

    for r1 in range(m):
        L1 = len(sol.tours[r1])
        if L1 < 2: continue

        for r2 in range(m):
            if r1 == r2: continue
            L2 = len(sol.tours[r2])

            for start in range(L1):
                for length in [1, 2, 3]:
                    if start + length > L1: continue

                    for pos in range(L2+1):

                        checks += 1
                        if max_checks and checks > max_checks: return improved

                        cand = ops.k_relocate(sol, r1, start, length, r2, pos)
                        if not cand: continue
                        cand.eval()

                        if first_improve:
                            if is_better(cand, sol):
                                return cand
                        else:
                            if improved is None or is_better(cand, improved):
                                improved = cand

    return improved


# ===================================================================
#  BVND 
# ===================================================================
NEIGHBORHOOD_CALLABLES  = [
    explore_intra_insertion,
    explore_inter_insertion,
    explore_intra_swap,
    explore_inter_swap,
    explore_two_opt,
    explore_two_opt_star,
    explore_cross_exchange,
    explore_k_relocate
]
NEIGHBORHOOD_NAMES = ['N1','N2','N3','N4','N5','N6','N7','N8']
NEIGHBORHOOD_FUNCS = NEIGHBORHOOD_CALLABLES

def b_vnd(
    sol: Solution,
    time_limit: Optional[float] = None,
    max_no_improve=1,
    first_improvement: bool = False,
    max_checks_per_neigh: int = None,
    
    usage_logger: dict = None,
    iteration: int = None,
    instance_name="unknown",
    order_mode="default"
) -> Solution:
    
    # neighborhood list 
    if order_mode == "reverse":
        neighborhoods = list(enumerate(NEIGHBORHOOD_FUNCS))[::-1]

    elif order_mode == "suggested":
        from gvns.utils import load_suggested_order

        order = load_suggested_order(instance_name)
        neighborhoods = [(i, NEIGHBORHOOD_FUNCS[i]) for i in order]

    elif order_mode == "random":
        rng = random.Random(iteration)  # for a different order for each iteration
        indices = list(range(len(NEIGHBORHOOD_FUNCS)))
        rng.shuffle(indices)
        neighborhoods = [(i, NEIGHBORHOOD_FUNCS[i]) for i in indices]

    else:  # "default"
        neighborhoods = list(enumerate(NEIGHBORHOOD_FUNCS))

    best = sol
    best.eval()

    start = time.time()
    no_improve = 0

    while True:
        if time_limit and (time.time() - start) > time_limit:
            break

        improved_any = False

        # sequential neighborhoods
        for idx, explorer in neighborhoods:

            if time_limit and (time.time() - start) > time_limit:
                break

            cand = explorer(
                best, 
                max_checks=max_checks_per_neigh, 
                first_improve=first_improvement
            )

            if cand and is_better(cand, best):
                best = cand
                improved_any = True

                # logging
                if usage_logger is not None and order_mode != "suggested":
                    usage_logger["counter"][idx] += 1
                    if iteration is not None:
                        usage_logger["events"].append({
                            "iteration": iteration,
                            "neighborhood_id": idx,
                            "neighborhood_name": NEIGHBORHOOD_NAMES[idx],
                            "F_before": best.Fval,
                            "F_after": cand.Fval,
                        })
                break  # restart from first neighborhood

        if not improved_any:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0
    
    return best

# ===================================================================
# helpers used for RL integration
# to run a single neighborhood (or list of neighborhoods) on a given solution
# ===================================================================
def rl_bvnd_step(sol: Solution,
                 agent,
                 obs,
                 neighborhoods,
                 first_improvement=False,
                 time_limit=None,
                 max_checks_per_neigh=None,
                 iteration: int = None,
                 usage_logger: dict = None,
                 verbose=False):
    """
    its one bvnd step using RL:
    - agent proposes neighborhoods
    - we try it and save it if it improves
    """
    action = agent.select_action(obs) # call a neighborhood

    cand = b_vnd_single_neigh(
        sol,
        neighborhoods[action],
        time_limit=time_limit,
        first_improvement=first_improvement,
        max_checks_per_neigh=max_checks_per_neigh
    )

    if cand and is_better(cand, sol):
        # logging 
        if usage_logger is not None:
            usage_logger["counter"][action] += 1
            if iteration is not None:
                usage_logger["events"].append({
                    "iteration": iteration,
                    "neighborhood_id": action,
                    "neighborhood_name": NEIGHBORHOOD_NAMES[action],
                    "F_before": sol.Fval,
                    "F_after": cand.Fval,
                })     
        return cand, action, True
    else:
        if verbose:
            print(f"[RL-BVND] Neighborhood {action} did not improve (F={sol.Fval})")
        return sol, action, False

def b_vnd_single_neigh(sol: Solution,
                       neighborhood,
                       time_limit=None,
                       max_no_improve=1,
                       first_improvement=False,
                       max_checks_per_neigh=None):
    """
    same logic as b_vnd, but restricted to ONE neighborhood.
    """
    assert callable(neighborhood), f"Neighborhood must be callable, got {type(neighborhood)}"

    best = sol
    best.eval()

    start = time.time()
    no_improve = 0

    while True:
        if time_limit and (time.time() - start) > time_limit:
            return best

        cand = neighborhood(
            best,
            max_checks=max_checks_per_neigh,
            first_improve=first_improvement
        )

        if cand and is_better(cand, best):
            best = cand
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                return best