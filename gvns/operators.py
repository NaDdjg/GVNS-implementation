# gvns/operators.py
from typing import List, Tuple, Optional
import random
from collections import Counter

from gvns.solution import Solution

# Toggle debug prints for operator diagnostics
DEBUG_OPERATORS = False

# ---------------------------------------------------------------------
# Helpers: multiset checks + diagnostics
# ---------------------------------------------------------------------
def multiset_of_tours(tours: List[List[int]]) -> Counter:
    c = Counter()
    for rt in tours:
        for v in rt:
            c[v] += 1
    return c

def is_multiset_equal(a: Counter, b: Counter) -> bool:
    return a == b

def diagnostic_print(op_name: str, before, after, details=""):
    if not DEBUG_OPERATORS:
        return
    print("[OPERATORS][DIAG]", op_name)
    print("  before total_visits:", sum(before.values()), "distinct:", len(before))
    print("  after  total_visits:", sum(after.values()), "distinct:", len(after))
    # show small samples
    print("  before sample:", {k:v for k,v in list(before.items())[:10]})
    print("  after  sample:", {k:v for k,v in list(after.items())[:10]})
    if details:
        print("  details:", details)

def safe_move_wrapper(sol: Solution, new_sol: Solution, op_name: str, extra_info: str="") -> Optional[Solution]:
    """
    Validate that the move preserved the multiset of visits.
    If invalid => print diagnostics and return None (move rejected).
    """
    before = multiset_of_tours(sol.tours)
    after = multiset_of_tours(new_sol.tours)
    if not is_multiset_equal(before, after):
        diagnostic_print(op_name, before, after, details=extra_info)
        return None
    return new_sol

# ---------------------------------------------------------------------
# Elementary operators (validated)
# ---------------------------------------------------------------------

def intra_insertion(sol: Solution, route_idx: int, from_pos: int, to_pos: int) -> Optional[Solution]:
    if from_pos == to_pos:
        return None
    if route_idx < 0 or route_idx >= len(sol.tours):
        return None
    route = sol.tours[route_idx]
    if from_pos < 0 or from_pos >= len(route):
        return None
    if to_pos < 0 or to_pos > len(route):
        return None

    new = sol.copy()
    rt = new.tours[route_idx]
    try:
        val = rt.pop(from_pos)
        insert_pos = to_pos if to_pos <= len(rt) else len(rt)
        rt.insert(insert_pos, val)
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] intra_insertion exception:", e)
        return None

    return safe_move_wrapper(sol, new, "intra_insertion", f"r{route_idx} {from_pos}->{to_pos}")

def inter_insertion(sol: Solution, from_route_idx: int, from_pos: int, to_route_idx: int, to_pos: int) -> Optional[Solution]:
    if from_route_idx == to_route_idx:
        return None
    if from_route_idx < 0 or to_route_idx < 0 or from_route_idx >= len(sol.tours) or to_route_idx >= len(sol.tours):
        return None
    if from_pos < 0 or from_pos >= len(sol.tours[from_route_idx]):
        return None

    new = sol.copy()
    try:
        val = new.tours[from_route_idx].pop(from_pos)
        insert_pos = to_pos if to_pos <= len(new.tours[to_route_idx]) else len(new.tours[to_route_idx])
        new.tours[to_route_idx].insert(insert_pos, val)
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] inter_insertion exception:", e)
        return None

    # Additional sync-safety check:
    inst = sol.instance
    cust = val
    if cust in getattr(inst, "sync_customers", set()):
        occs = [rt.count(cust) for rt in new.tours]
        total_occ = sum(occs)
        req = inst.sync_requirements.get(cust, 2)
        # If a single route now contains all occurrences (i.e., all required occurrences ended up on a single route),
        # then the synchronization requirement would be violated (we need them on different vehicles).
        for cnt in occs:
            if cnt >= req and cnt == total_occ and total_occ >= req:
                # diagnostic
                diagnostic_print("inter_insertion_sync_violation",
                                 multiset_of_tours(sol.tours), multiset_of_tours(new.tours),
                                 details=f"cust {cust} occs_by_route={occs}")
                return None

    return safe_move_wrapper(sol, new, "inter_insertion", f"{from_route_idx}:{from_pos} -> {to_route_idx}:{to_pos}")

def intra_swap(sol: Solution, route_idx: int, pos_a: int, pos_b: int) -> Optional[Solution]:
    if route_idx < 0 or route_idx >= len(sol.tours):
        return None
    rt = sol.tours[route_idx]
    if pos_a == pos_b:
        return None
    if pos_a < 0 or pos_b < 0 or pos_a >= len(rt) or pos_b >= len(rt):
        return None
    new = sol.copy()
    try:
        new.tours[route_idx][pos_a], new.tours[route_idx][pos_b] = new.tours[route_idx][pos_b], new.tours[route_idx][pos_a]
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] intra_swap exception:", e)
        return None
    return safe_move_wrapper(sol, new, "intra_swap", f"r{route_idx} p{pos_a}<->{pos_b}")

def inter_swap(sol: Solution, r1: int, p1: int, r2: int, p2: int) -> Optional[Solution]:
    if r1 == r2:
        return None
    if r1 < 0 or r2 < 0 or r1 >= len(sol.tours) or r2 >= len(sol.tours):
        return None
    if p1 < 0 or p2 < 0 or p1 >= len(sol.tours[r1]) or p2 >= len(sol.tours[r2]):
        return None

    new = sol.copy()
    try:
        new.tours[r1][p1], new.tours[r2][p2] = new.tours[r2][p2], new.tours[r1][p1]
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] inter_swap exception:", e)
        return None

    # sync-safety: ensure we didn't end up with all required occurrences in the same route
    inst = sol.instance
    c1 = new.tours[r1][p1]
    c2 = new.tours[r2][p2]
    for cust in (c1, c2):
        if cust in getattr(inst, "sync_customers", set()):
            occs = [rt.count(cust) for rt in new.tours]
            total_occ = sum(occs)
            req = inst.sync_requirements.get(cust, 2)
            for cnt in occs:
                if cnt >= req and cnt == total_occ and total_occ >= req:
                    diagnostic_print("inter_swap_sync_violation",
                                     multiset_of_tours(sol.tours), multiset_of_tours(new.tours),
                                     details=f"cust {cust} occs_by_route={occs}")
                    return None

    return safe_move_wrapper(sol, new, "inter_swap", f"r{r1}p{p1}<->{r2}p{p2}")

# ---------------------------------------------------------------------
# Randomized/shaking utilities (use validated operators)
# ---------------------------------------------------------------------
def random_shake(sol: Solution, k: int, insert_prob: float = 0.5) -> Solution:
    new = sol.copy()
    m = len(new.tours)
    attempts = 0
    for _ in range(k):
        attempts += 1
        if attempts > max(1000, k*10):
            break
        r = random.random()
        if r < insert_prob:
            # insertion moves
            non_empty = [i for i in range(m) if len(new.tours[i]) > 0]
            if not non_empty:
                continue
            a = random.choice(non_empty)
            # 50% intra if route length>1 else inter
            if len(new.tours[a]) > 1 and random.random() < 0.5:
                from_pos = random.randrange(len(new.tours[a]))
                to_pos = random.randrange(len(new.tours[a]) + 1)
                candidate = intra_insertion(new, a, from_pos, to_pos)
                if candidate:
                    new = candidate
            else:
                # inter-insertion
                b = random.randrange(m)
                if b == a or len(new.tours[a]) == 0:
                    continue
                from_pos = random.randrange(len(new.tours[a]))
                to_pos = random.randrange(len(new.tours[b]) + 1)
                candidate = inter_insertion(new, a, from_pos, b, to_pos)
                if candidate:
                    new = candidate
        else:
            # swap type
            if random.random() < 0.5:
                idxs = [i for i in range(m) if len(new.tours[i]) > 1]
                if not idxs:
                    continue
                r_idx = random.choice(idxs)
                p1 = random.randrange(len(new.tours[r_idx]))
                p2 = random.randrange(len(new.tours[r_idx]))
                candidate = intra_swap(new, r_idx, p1, p2)
                if candidate:
                    new = candidate
            else:
                non_empty = [i for i in range(m) if len(new.tours[i]) > 0]
                if len(non_empty) < 2:
                    continue
                a, b = random.sample(non_empty, 2)
                p1 = random.randrange(len(new.tours[a]))
                p2 = random.randrange(len(new.tours[b]))
                candidate = inter_swap(new, a, p1, b, p2)
                if candidate:
                    new = candidate

    return new
