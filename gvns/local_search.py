from typing import List, Optional
import random
import time

from gvns.solution import Solution
import gvns.operators as ops

"""
the logic of first improvement will make thigs faster since it stops on the frist useful move, 
it will keep a diversification with more small improvements 
"""
def explore_N1_first_improvement(sol: Solution, max_tries: int = 500, sample: bool = True) -> Optional[Solution]:
    """
    search intra-insertion moves (N1) trying to find a solution with strictly better Fval.
    returns the improved solution or None.
    """
    m = len(sol.tours)
    base_val = sol.Fval
    # enumerate or sample routes and positions
    tries = 0
    route_indices = list(range(m))
    if sample:
        random.shuffle(route_indices)
    for r in route_indices:
        ln = len(sol.tours[r])
        if ln < 2:
            continue
        positions = list(range(ln))
        if sample:
            random.shuffle(positions)
        for i in positions:
            # possible target insert positions
            targets = list(range(ln))
            if sample:
                random.shuffle(targets)
            for t in targets:
                tries += 1
                if tries > max_tries:
                    return None
                new = ops.intra_insertion(sol, r, i, t)
                if new is None:
                    continue
                new.eval()
                if new.Fval < base_val - 1e-9:
                    return new
    return None

def explore_N2_first_improvement(sol: Solution, max_tries: int = 2000, sample: bool = True) -> Optional[Solution]:
    """inter-insertion"""
    m = len(sol.tours)
    base_val = sol.Fval
    tries = 0
    route_pairs = [(i,j) for i in range(m) for j in range(m) if i != j]
    if sample:
        random.shuffle(route_pairs)
    for (a,b) in route_pairs:
        if len(sol.tours[a]) == 0:
            continue
        positions = list(range(len(sol.tours[a])))
        if sample:
            random.shuffle(positions)
        for pos in positions:
            targets = list(range(len(sol.tours[b]) + 1))
            if sample:
                random.shuffle(targets)
            for t in targets:
                tries += 1
                if tries > max_tries:
                    return None
                new = ops.inter_insertion(sol, a, pos, b, t)
                if new is None:
                    continue
                new.eval()
                if new.Fval < base_val - 1e-9:
                    return new
    return None

def explore_N3_first_improvement(sol: Solution, max_tries: int = 1000, sample: bool = True) -> Optional[Solution]:
    # intra-swap
    m = len(sol.tours)
    base_val = sol.Fval
    tries = 0
    route_indices = list(range(m))
    if sample:
        random.shuffle(route_indices)
    for r in route_indices:
        ln = len(sol.tours[r])
        if ln < 2:
            continue
        pos_list = list(range(ln))
        if sample:
            random.shuffle(pos_list)
        for a in pos_list:
            for b in pos_list:
                tries += 1
                if tries > max_tries:
                    return None
                new = ops.intra_swap(sol, r, a, b)
                if new is None:
                    continue
                new.eval()
                if new.Fval < base_val - 1e-9:
                    return new
    return None

def explore_N4_first_improvement(sol: Solution, max_tries: int = 2000, sample: bool = True) -> Optional[Solution]:
    # inter-swap
    m = len(sol.tours)
    base_val = sol.Fval
    tries = 0
    pairs = [(i,j) for i in range(m) for j in range(m) if i < j]
    if sample:
        random.shuffle(pairs)
    for (a,b) in pairs:
        if len(sol.tours[a]) == 0 or len(sol.tours[b]) == 0:
            continue
        p1s = list(range(len(sol.tours[a])))
        p2s = list(range(len(sol.tours[b])))
        if sample:
            random.shuffle(p1s); random.shuffle(p2s)
        for p1 in p1s:
            for p2 in p2s:
                tries += 1
                if tries > max_tries:
                    return None
                new = ops.inter_swap(sol, a, p1, b, p2)
                if new is None:
                    continue
                new.eval()
                if new.Fval < base_val - 1e-9:
                    return new
    return None

# neighbors 
NEIGH_EXPLORERS = {
    "N1": explore_N1_first_improvement,
    "N2": explore_N2_first_improvement,
    "N3": explore_N3_first_improvement,
    "N4": explore_N4_first_improvement
}

def b_vnd(sol: Solution,
          neighborhoods: List[str] = ["N1", "N2", "N3", "N4"],
          max_no_improve_iters: int = 10,
          time_limit: Optional[float] = None) -> Solution:
    """
    B-VND logic 
    - neighborhoods: ordered list (N1..N4)
    - iterate neighborhoods sequentially, when an improvement is found we restart from first neighborhood
    - stop after max_no_improve_iters consecutive outer loops without improvement or time_limit seconds
    """
    best = sol
    best.eval()
    start = time.time()
    no_improve = 0
    outer_iter = 0

    while True:
        if time_limit and (time.time() - start) > time_limit:
            break
        improved = False
        for nb in neighborhoods:
            explorer = NEIGH_EXPLORERS.get(nb)
            if explorer is None:
                continue
            new = explorer(best)
            if new is not None:
                new.eval()
                if new.Fval < best.Fval - 1e-9:
                    best = new
                    improved = True
                    break  # restart from first neighborhood
        outer_iter += 1
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve_iters:
                break
    return best
