from typing import List, Optional
import random
import time

from gvns.solution import Solution
import gvns.operators as ops

def _route_len(sol: Solution, r: int) -> int:
    """Return length of route r treating missing routes as empty"""
    return len(sol.tours[r]) if r < len(sol.tours) else 0

def explore_N1_first_improvement(sol: Solution, max_tries: int = 500, sample: bool = True) -> Optional[Solution]:
    """intra-insertion (N1) - first improvement"""
    m = sol.instance.m
    base_val = sol.Fval
    tries = 0
    route_indices = list(range(m))
    if sample:
        random.shuffle(route_indices)
    for r in route_indices:
        ln = _route_len(sol, r)
        if ln < 2:
            continue
        positions = list(range(ln))
        if sample:
            random.shuffle(positions)
        targets = list(range(ln + 1))  # include append
        for i in positions:
            t_list = list(targets)
            if sample:
                random.shuffle(t_list)
            for t in t_list:
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
    """inter-insertion (N2) - first improvement"""
    m = sol.instance.m
    base_val = sol.Fval
    tries = 0
    route_pairs = [(i, j) for i in range(m) for j in range(m) if i != j]
    if sample:
        random.shuffle(route_pairs)
    for (a, b) in route_pairs:
        la = _route_len(sol, a)
        if la == 0:
            continue
        positions = list(range(la))
        if sample:
            random.shuffle(positions)
        for pos in positions:
            lb = _route_len(sol, b)
            targets = list(range(lb + 1))
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
    """intra-swap (N3) - first improvement"""
    m = sol.instance.m
    base_val = sol.Fval
    tries = 0
    route_indices = list(range(m))
    if sample:
        random.shuffle(route_indices)
    for r in route_indices:
        ln = _route_len(sol, r)
        if ln < 2:
            continue
        pos_list = list(range(ln))
        if sample:
            random.shuffle(pos_list)
        for a in pos_list:
            for b in pos_list:
                if a == b:
                    continue
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
    """inter-swap (N4) - first improvement"""
    m = sol.instance.m
    base_val = sol.Fval
    tries = 0
    pairs = [(i, j) for i in range(m) for j in range(m) if i < j]
    if sample:
        random.shuffle(pairs)
    for (a, b) in pairs:
        la = _route_len(sol, a)
        lb = _route_len(sol, b)
        if la == 0 or lb == 0:
            continue
        p1s = list(range(la))
        p2s = list(range(lb))
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

NEIGH_EXPLORERS = {
    "N1": explore_N1_first_improvement,
    "N2": explore_N2_first_improvement,
    "N3": explore_N3_first_improvement,
    "N4": explore_N4_first_improvement
}

def better_by_penalties_and_F(old_eval, new_eval, eps_f=1e-9):
    """Lexicographic comparator: PenS, PenT, PenH, then F."""
    oS = old_eval.get("penS", getattr(old_eval, "penS", 0.0))
    nS = new_eval.get("penS", getattr(new_eval, "penS", 0.0))
    if nS < oS - 1e-12:
        return True
    if abs(nS - oS) > 1e-12:
        return False
    oT = old_eval.get("penT", getattr(old_eval, "penT", 0.0))
    nT = new_eval.get("penT", getattr(new_eval, "penT", 0.0))
    if nT < oT - 1e-12:
        return True
    if abs(nT - oT) > 1e-12:
        return False
    oH = old_eval.get("penH", getattr(old_eval, "penH", 0.0))
    nH = new_eval.get("penH", getattr(new_eval, "penH", 0.0))
    if nH < oH - 1e-12:
        return True
    if abs(nH - oH) > 1e-12:
        return False
    oF = old_eval.get("F", getattr(old_eval, "F", old_eval.get("Fval", 0.0)))
    nF = new_eval.get("F", getattr(new_eval, "F", new_eval.get("Fval", 0.0)))
    return nF < oF - eps_f

# VNDt inner loop (first-improvement scanning) 
def _vnd_first_improvement(base_sol: Solution,
                           neighborhoods: List[str],
                           rl_i_pass: int = 500,
                           time_limit: Optional[float] = None) -> Solution:
    explorers = {
        "N1": globals().get("explore_N1_first_improvement"),
        "N2": globals().get("explore_N2_first_improvement"),
        "N3": globals().get("explore_N3_first_improvement"),
        "N4": globals().get("explore_N4_first_improvement"),
    }

    best = base_sol
    best.eval()
    start = time.time()

    while True:
        improved_this_round = False
        for nb in neighborhoods:
            if time_limit and (time.time() - start) > time_limit:
                return best
            explorer = explorers.get(nb)
            if explorer is None:
                continue
            try:
                cand = explorer(best, max_tries=rl_i_pass, sample=True)
            except TypeError:
                cand = explorer(best)
            if cand is None:
                continue
            cand.eval()
            if better_by_penalties_and_F({"penS": best.penS, "penT": best.penT, "penH": best.penH, "F": best.Fval},
                                         {"penS": cand.penS, "penT": cand.penT, "penH": cand.penH, "F": cand.Fval}):
                best = cand
                improved_this_round = True
                break
        if not improved_this_round:
            break
    best.eval()
    return best

def b_vnd(sol: Solution,
          neighborhoods: List[str] = ["N1", "N2", "N3", "N4"],
          max_no_improve_iters: int = 10,
          time_limit: Optional[float] = None,
          rl_p_pass: int = 3,    # RL_P (outer perturbation+VND cycles)
          rl_i_pass: int = 400,  # RL_I (per-neighborhood trial budget)
          l: int = 6,            # number of small random perturbations per pass (l)
          k_shake_small: int = 1 # k for ops.random_shake inside perturbation
         ) -> Solution:
    best = sol
    best.eval()
    start = time.time()
    no_improve_outer = 0

    # sensible default if user passed <=0
    if rl_i_pass <= 0:
        rl_i_pass = max(500, int((sum(len(t) for t in sol.tours) or 1) * 4))

    while True:
        if time_limit and (time.time() - start) > time_limit:
            break
        improved_in_outer = False

        # RL_P loop (outer perturbation+VND cycles)
        for pass_i in range(rl_p_pass):
            if time_limit and (time.time() - start) > time_limit:
                break

            candidate = best.copy()
            # apply l small perturbations sequentially (small shakes)
            for _ in range(l):
                candidate = ops.random_shake(candidate, k=k_shake_small, insert_prob=0.6)
                # keep penalties consistent
                candidate.w1 = best.w1; candidate.w2 = best.w2; candidate.alpha = best.alpha; candidate.M = best.M

            # run inner VND (per-neighborhood budget = rl_i_pass)
            time_left = None
            if time_limit:
                time_left = max(0.01, time_limit - (time.time() - start))
            improved_candidate = _vnd_first_improvement(candidate, neighborhoods, rl_i_pass, time_left)
            improved_candidate.eval()

            best_eval = {"penS": best.penS, "penT": best.penT, "penH": best.penH, "F": best.Fval}
            cand_eval = {"penS": improved_candidate.penS, "penT": improved_candidate.penT,
                         "penH": improved_candidate.penH, "F": improved_candidate.Fval}
            if better_by_penalties_and_F(best_eval, cand_eval):
                best = improved_candidate
                improved_in_outer = True
                break

        if improved_in_outer:
            no_improve_outer = 0
            continue
        else:
            no_improve_outer += 1
            if no_improve_outer >= max_no_improve_iters:
                break

    best.eval()
    return best

# def b_vnd(sol: Solution,
#           neighborhoods: List[str] = ["N1", "N2", "N3", "N4"],
#           max_no_improve_iters: int = 10,
#           time_limit: Optional[float] = None) -> Solution:
#     """
#     B-VND logic 
#     - neighborhoods: ordered list (N1..N4)
#     - iterate neighborhoods sequentially, when an improvement is found we restart from first neighborhood
#     - stop after max_no_improve_iters consecutive outer loops without improvement or time_limit seconds
#     """
#     best = sol
#     best.eval()
#     start = time.time()
#     no_improve = 0
#     outer_iter = 0

#     while True:
#         if time_limit and (time.time() - start) > time_limit:
#             break
#         improved = False
#         for nb in neighborhoods:
#             explorer = NEIGH_EXPLORERS.get(nb)
#             if explorer is None:
#                 continue
#             new = explorer(best)
#             if new is not None:
#                 new.eval()
#                 if new.Fval < best.Fval - 1e-9:
#                     best = new
#                     improved = True
#                     break  # restart from first neighborhood
#         outer_iter += 1
#         if improved:
#             no_improve = 0
#         else:
#             no_improve += 1
#             if no_improve >= max_no_improve_iters:
#                 break
#     return best