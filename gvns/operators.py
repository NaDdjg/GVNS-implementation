from typing import List, Tuple, Optional
import random
from collections import Counter

from gvns.solution import Solution

# debug prints for operator diagnostics
DEBUG_OPERATORS = False

# helpers: multiset checks + diagnostics
def multiset_of_tours(tours: List[List[int]]) -> Counter:
    """
    returns a counter of totsl occurences of each customer across all tours
    used to check that moves preserve the multiset of visits (no lost/duplicated visits)
    """
    c = Counter()
    for rt in tours:
        for v in rt:
            c[v] += 1
    return c

def diagnostic_print(op_name: str, before, after, details=""):
    if not DEBUG_OPERATORS:
        return
    print("[OPERATORS][DIAG]", op_name)
    print("  before total_visits:", sum(before.values()), "distinct:", len(before))
    print("  after  total_visits:", sum(after.values()), "distinct:", len(after))
    if details:
        print("  details:", details)

def safe_tt(TT, i, j):
    """safe travel time access - mirrors solution.safe_tt behavior"""
    try:
        if isinstance(TT, dict):
            return TT.get(i, {}).get(j, 0.0)
        else:
            return TT[i][j]
    except Exception:
        return 0.0
    
def _ensure_m_tours(tours: List[List[int]], m: int):
    # modify list in-place to have exactly m inner-lists
    if len(tours) < m:
        tours.extend([[] for _ in range(m - len(tours))])
    elif len(tours) > m:
        # do not shrink unless necessary; when called, shrink to m explicitly
        del tours[m:]

def safe_move_wrapper(sol: Solution, new_sol: Solution, op_name: str, extra_info: str="") -> Optional[Solution]:
    """
    allow temporary violations (do not check multisets).
    This function pads/truncates new_sol.tours to m and returns new_sol.
    Use DEBUG_OPERATORS to optionally show multiset differences.
    """
    inst = sol.instance
    m = inst.m

    # pad/truncate inplace
    if len(new_sol.tours) < m:
        new_sol.tours.extend([[] for _ in range(m - len(new_sol.tours))])
    elif len(new_sol.tours) > m:
        new_sol.tours = new_sol.tours[:m]

    if DEBUG_OPERATORS:
        before = multiset_of_tours([list(r) for r in sol.tours])
        after = multiset_of_tours([list(r) for r in new_sol.tours])
        if before != after:
            diagnostic_print(op_name, before, after, details=extra_info)

    return new_sol

#  operators : they are are allowed to move occurrences between routes but will refuse any move that would put all required occurrences for a synced customer on the same vehicle
def intra_insertion(sol: Solution, route_idx: int, from_pos: int, to_pos: int) -> Optional[Solution]:
    """
    move an item inside the same route from from_pos to to_pos
    returns None if invalid indices or positions equal 
    """
    if from_pos == to_pos:
        return None

    m = sol.instance.m
    if route_idx < 0 or route_idx >= m:
        return None
    
    route = sol.tours[route_idx] if route_idx < len(sol.tours) else []
    if from_pos < 0 or from_pos >= len(route):
        return None
    if to_pos < 0 or to_pos > len(route):
        return None

    new = sol.copy()
    _ensure_m_tours(new.tours, m)
    try:
        val = new.tours[route_idx].pop(from_pos)
        insert_pos = to_pos if to_pos <= len(new.tours[route_idx]) else len(new.tours[route_idx])
        new.tours[route_idx].insert(insert_pos, val)
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] intra_insertion exception:", e)
        return None

    return safe_move_wrapper(sol, new, "intra_insertion", f"r{route_idx} {from_pos}->{to_pos}")

def inter_insertion(sol: Solution, from_route_idx: int, from_pos: int, to_route_idx: int, to_pos: int) -> Optional[Solution]:
    """
    remove an element from route A and insert it into route B
    added a check for synchronized elements: 
        - if the customer is synchronized, it checks occs = [rt.count(cust) for rt in new.tours] and total_occ
        - If after the move all required occurrences end up on a single vehicle (route) it rejects the move
    it will prevents putting both occurrences on the same vehicle
    """
    if from_route_idx == to_route_idx:
        return None
    
    m = sol.instance.m
    if from_route_idx < 0 or to_route_idx < 0 or from_route_idx >= m or to_route_idx >= m:
        return None
    # ensure source route exists
    if from_route_idx >= len(sol.tours) or from_pos < 0 or from_pos >= len(sol.tours[from_route_idx]):
        return None

    new = sol.copy()
    _ensure_m_tours(new.tours, m)
    try:
        val = new.tours[from_route_idx].pop(from_pos)
        insert_pos = to_pos if to_pos <= len(new.tours[to_route_idx]) else len(new.tours[to_route_idx])
        new.tours[to_route_idx].insert(insert_pos, val)
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] inter_insertion exception:", e)
        return None

    # synch part 
    # inst = sol.instance
    # cust = val
    # if cust in getattr(inst, "sync_customers", set()):
    #     occs = [rt.count(cust) for rt in new.tours]
    #     total_occ = sum(occs)
    #     req = inst.sync_requirements.get(cust, 2)
    #     # if a single route now contains all occurrences (so all required occurrences ended up on a single route),
    #     # then the synchronization requirement would be violated (we need them on different vehicles) so the move is changed
    #     for cnt in occs:
    #         if cnt >= req and cnt == total_occ and total_occ >= req:
    #             # diagnostic
    #             diagnostic_print("inter_insertion_sync_violation",
    #                              multiset_of_tours(sol.tours), multiset_of_tours(new.tours),
    #                              details=f"cust {cust} occs_by_route={occs}")
    #             return None

    return safe_move_wrapper(sol, new, "inter_insertion", f"{from_route_idx}:{from_pos} -> {to_route_idx}:{to_pos}")

def intra_swap(sol: Solution, route_idx: int, pos_a: int, pos_b: int) -> Optional[Solution]:
    """
    swap two positions inside a route
    returns None if invalid indices
    """
    m = sol.instance.m
    if route_idx < 0 or route_idx >= m:
        return None
    rt = sol.tours[route_idx] if route_idx < len(sol.tours) else []
    if pos_a == pos_b:
        return None
    if pos_a < 0 or pos_b < 0 or pos_a >= len(rt) or pos_b >= len(rt):
        return None
    
    new = sol.copy()
    _ensure_m_tours(new.tours, m)
    try:
        new.tours[route_idx][pos_a], new.tours[route_idx][pos_b] = new.tours[route_idx][pos_b], new.tours[route_idx][pos_a]
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] intra_swap exception:", e)
        return None
    return safe_move_wrapper(sol, new, "intra_swap", f"r{route_idx} p{pos_a}<->{pos_b}")

def inter_swap(sol: Solution, r1: int, p1: int, r2: int, p2: int) -> Optional[Solution]:
    """
    swap an item from route r1 with an item from r2
    added safety for synchronized ones: for both swapped customers it recomputes occurrences per route 
        and rejects if the swap concentrates all required occs on a single route
    """
    m = sol.instance.m
    if r1 == r2:
        return None
    if r1 < 0 or r2 < 0 or r1 >= m or r2 >= m:
        return None
    if r1 >= len(sol.tours) or r2 >= len(sol.tours):
        return None
    if p1 < 0 or p2 < 0 or p1 >= len(sol.tours[r1]) or p2 >= len(sol.tours[r2]):
        return None

    new = sol.copy()
    _ensure_m_tours(new.tours, m)
    try:
        new.tours[r1][p1], new.tours[r2][p2] = new.tours[r2][p2], new.tours[r1][p1]
    except Exception as e:
        if DEBUG_OPERATORS:
            print("[OPERATORS] inter_swap exception:", e)
        return None

    # synch part 
    # inst = sol.instance
    # c1 = new.tours[r1][p1]
    # c2 = new.tours[r2][p2]
    # for cust in (c1, c2):
    #     if cust in getattr(inst, "sync_customers", set()):
    #         occs = [rt.count(cust) for rt in new.tours]
    #         total_occ = sum(occs)
    #         req = inst.sync_requirements.get(cust, 2)
    #         for cnt in occs:
    #             if cnt >= req and cnt == total_occ and total_occ >= req:
    #                 diagnostic_print("inter_swap_sync_violation",
    #                                  multiset_of_tours(sol.tours), multiset_of_tours(new.tours),
    #                                  details=f"cust {cust} occs_by_route={occs}")
    #                 return None

    return safe_move_wrapper(sol, new, "inter_swap", f"r{r1}p{p1}<->{r2}p{p2}")

# def k_relocate(sol: Solution, r_from: int, start: int, length: int, r_to: int, pos: int):
#     """
#     Move a segment of length `length` from route r_from[start:start+length]
#     into route r_to at position pos.
#     Preserves multiset exactly.
#     Sync constraints enforced: no segment may create both sync occurrences
#     on same vehicle.
#     """
#     m = sol.instance.m
#     if r_from < 0 or r_to < 0 or r_from >= m or r_to >= m:
#         return None
#     if r_from >= len(sol.tours) or r_to >= len(sol.tours):
#         return None
    
#     route_from = sol.tours[r_from]
#     route_to = sol.tours[r_to]

#     if start < 0 or start >= len(route_from):
#         return None
#     if start + length > len(route_from):
#         return None
#     if pos < 0 or pos > len(route_to):
#         return None
    
#     segment = route_from[start:start+length]

#     # sync constraint: segment cannot contain the only copy of a sync
#     inst = sol.instance
#     sync_custs = inst.sync_customers

#     for cust in segment:
#         if cust in sync_custs:
#             # check occurrences after move
#             occs_before = [rt.count(cust) for rt in sol.tours]
#             # simulate move
#             occs_after = occs_before[:]
#             occs_after[r_from] -= 1
#             occs_after[r_to] += 1
#             req = inst.sync_requirements.get(cust, 2)
#             # if all req copies end up on r_to → reject
#             if occs_after[r_to] >= req and sum(occs_after) == occs_after[r_to]:
#                 return None
    
#     new = sol.copy()
#     _ensure_m_tours(new.tours, m)

#     seg = new.tours[r_from][start:start+length]
#     del new.tours[r_from][start:start+length]
#     # adjust pos if same route and the removed block is before pos
#     if r_from == r_to and pos > start:
#         pos -= length
#     new.tours[r_to][pos:pos] = seg

#     return safe_move_wrapper(sol, new, "k_relocate",
#                              f"{r_from}[{start}:{start+length}] -> {r_to} at {pos}")

def k_relocate(sol: Solution, r_from: int, start: int, length: int, r_to: int, pos: int) -> Optional[Solution]:
    """
    Move a contiguous segment of `length` from r_from[start:start+length] into r_to at position pos.
    Option A: allowed to temporarily break multiplicities.
    """
    m = sol.instance.m
    if r_from < 0 or r_to < 0 or r_from >= m or r_to >= m:
        return None
    if r_from >= len(sol.tours):
        return None
    if start < 0 or start >= len(sol.tours[r_from]):
        return None
    if length <= 0 or start + length > len(sol.tours[r_from]):
        return None
    if pos < 0:
        return None
    new = sol.copy()
    _ensure_m_tours(new.tours, m)

    segment = new.tours[r_from][start:start+length]
    del new.tours[r_from][start:start+length]
    if r_from == r_to and pos > start:
        pos -= length
    if pos > len(new.tours[r_to]):
        pos = len(new.tours[r_to])
    new.tours[r_to][pos:pos] = segment

    return safe_move_wrapper(sol, new, "k_relocate", f"{r_from}[{start}:{start+length}] -> {r_to}@{pos}")


def two_opt(sol: Solution, r: int, i: int, j: int):
    """
    Reverse segment [i..j] inside route r.
    """
    m = sol.instance.m
    if r < 0 or r >= m:
        return None
    rt = sol.tours[r]
    if i < 0 or j < 0 or i >= len(rt) or j >= len(rt) or i >= j:
        return None
    
    new = sol.copy()
    _ensure_m_tours(new.tours, m)

    new.tours[r][i:j+1] = reversed(new.tours[r][i:j+1])

    return safe_move_wrapper(sol, new, "2-opt", f"r{r} {i}-{j}")

def two_opt_star(sol: Solution, r1: int, i: int, r2: int, j: int):
    """
    2-opt* between two routes: swap suffixes.
    """
    if r1 == r2:
        return None
    m = sol.instance.m
    if r1 < 0 or r1 >= m or r2 < 0 or r2 >= m:
        return None
    
    rt1 = sol.tours[r1]
    rt2 = sol.tours[r2]
    
    if i <= 0 or j <= 0:
        return None
    if i > len(rt1) or j > len(rt2):
        return None

    new = sol.copy()
    _ensure_m_tours(new.tours, m)
    A = new.tours[r1]
    B = new.tours[r2]

    newA = A[:i] + B[j:]
    newB = B[:j] + A[i:]

    new.tours[r1] = newA
    new.tours[r2] = newB

    # check sync constraints → reject if violated
    # inst = sol.instance
    # for cust in inst.sync_customers:
    #     req = inst.sync_requirements.get(cust, 2)
    #     occs = [rt.count(cust) for rt in new.tours]
    #     if any(cnt >= req and cnt == sum(occs) for cnt in occs):
    #         return None

    return safe_move_wrapper(sol, new, "2-opt*", f"{r1}:{i} <-> {r2}:{j}")

def cross_exchange(sol: Solution, r1: int, a1: int, b1: int, r2: int, a2: int, b2: int) -> Optional[Solution]:
    """
    Exchange segments A[a1:b1] with B[a2:b2]. Inclusive-exclusive: [a1:b1), [a2:b2)
    """
    m = sol.instance.m
    if r1 < 0 or r2 < 0 or r1 >= m or r2 >= m:
        return None
    if r1 >= len(sol.tours) or r2 >= len(sol.tours):
        return None
    A = sol.tours[r1]
    B = sol.tours[r2]
    if not (0 <= a1 < b1 <= len(A)) or not (0 <= a2 < b2 <= len(B)):
        return None

    new = sol.copy()
    _ensure_m_tours(new.tours, m)
    segA = new.tours[r1][a1:b1]
    segB = new.tours[r2][a2:b2]
    new.tours[r1][a1:b1] = segB
    new.tours[r2][a2:b2] = segA

    return safe_move_wrapper(sol, new, "cross_exchange", f"r{r1}[{a1}:{b1}] <-> r{r2}[{a2}:{b2}]")

# def relocate_sync_pair(sol: Solution, cust: int):
#     """
#     Move both occurrences of a sync customer to new vehicles/positions
#     keeping them on different vehicles.
#     """
#     inst = sol.instance
#     if cust not in inst.sync_customers:
#         return None
    
#     # find occurrences
#     occ = [(r,pos) for r,rt in enumerate(sol.tours) 
#                     for pos,v in enumerate(rt) if v == cust]
#     if len(occ) != 2:
#         return None
    
#     (r1,p1), (r2,p2) = occ

#     # pick two new vehicles (not equal)
#     m = inst.m
#     all_veh = list(range(m))
#     random.shuffle(all_veh)

#     for v1 in all_veh:
#         for v2 in all_veh:
#             if v1 == v2:
#                 continue
#             new = sol.copy()
#             _ensure_m_tours(new.tours, m)
#             # remove original occurrences
#             new.tours[r1].pop(p1)
#             # adjust second position depending on removal order
#             if r1 == r2 and p2 > p1:
#                 p2 -= 1
#             new.tours[r2].pop(p2)
#             # insert in new positions
#             pos1 = random.randrange(len(new.tours[v1])+1)
#             pos2 = random.randrange(len(new.tours[v2])+1)
#             new.tours[v1].insert(pos1, cust)
#             new.tours[v2].insert(pos2, cust)

#             # check ok
#             return safe_move_wrapper(sol, new, "relocate_sync_pair",
#                                      f"{cust} {r1},{p1} & {r2},{p2} -> {v1},{pos1} & {v2},{pos2}")

#     return None
def relocate_sync_pair(sol: Solution, cust: int, attempts: int = 20) -> Optional[Solution]:
    """
    Remove both occurrences of a synchronized customer and reinsert them randomly
    into two (distinct) vehicles at (random) positions. This helps realign synchronization.
    Attempts up to `attempts` random trials and returns the first valid candidate (does not enforce feasibility).
    """
    inst = sol.instance
    if cust not in getattr(inst, "sync_customers", set()):
        return None

    occ = [(r, pos) for r, rt in enumerate(sol.tours) for pos, v in enumerate(rt) if v == cust]
    if len(occ) < 1:
        return None

    # if more than two occurrences exist, handle all occurrences: remove them then reinsert req times
    req = inst.sync_requirements.get(cust, 2)
    # create base copy
    for _ in range(attempts):
        new = sol.copy()
        _ensure_m_tours(new.tours, inst.m)
        # remove up to all occurrences
        removed = 0
        for r, pos in sorted(occ, key=lambda x: (x[0], -x[1]), reverse=True):
            try:
                # if route may have shifted due to previous removals, attempt safe removal
                if r < len(new.tours) and (pos < len(new.tours[r]) and new.tours[r][pos] == cust):
                    new.tours[r].pop(pos)
                    removed += 1
                else:
                    # try to remove any occurrence from that route
                    if r < len(new.tours) and cust in new.tours[r]:
                        new.tours[r].remove(cust)
                        removed += 1
            except Exception:
                pass

        # now reinsert req occurrences into random distinct vehicles
        vehicles = list(range(inst.m))
        random.shuffle(vehicles)
        chosen = []
        for v in vehicles:
            if len(chosen) >= req:
                break
            chosen.append(v)

        if len(chosen) < req:
            continue

        for idx, v in enumerate(chosen):
            pos = random.randrange(len(new.tours[v]) + 1)
            new.tours[v].insert(pos, cust)

        return safe_move_wrapper(sol, new, "relocate_sync_pair", f"cust {cust}")

    return None

# shaking utilities 
# def random_shake(sol: Solution, k: int, insert_prob: float = 0.5) -> Solution:
#     new = sol.copy()
#     m = sol.instance.m
#     _ensure_m_tours(new.tours, m)
#     attempts = 0
#     for _ in range(k):
#         attempts += 1
#         if attempts > max(1000, k*10):
#             break
#         r = random.random()
#         if r < insert_prob:
#             # insertion moves
#             while len(new.tours) < m:
#                 new.tours.append([])
#             if len(new.tours) > m:
#                 new.tours = new.tours[:m]
#             non_empty = [i for i in range(m) if len(new.tours[i]) > 0]
#             if not non_empty:
#                 continue
#             a = random.choice(non_empty)
#             # 50% intra if route length>1 else inter
#             if len(new.tours[a]) > 1 and random.random() < 0.5:
#                 from_pos = random.randrange(len(new.tours[a]))
#                 to_pos = random.randrange(len(new.tours[a]) + 1)
#                 candidate = intra_insertion(new, a, from_pos, to_pos)
#                 if candidate:
#                     new = candidate
#             else:
#                 # inter-insertion
#                 b = random.randrange(m)
#                 if b == a or len(new.tours[a]) == 0:
#                     continue
#                 from_pos = random.randrange(len(new.tours[a]))
#                 to_pos = random.randrange(len(new.tours[b]) + 1)
#                 candidate = inter_insertion(new, a, from_pos, b, to_pos)
#                 if candidate:
#                     new = candidate
#         else:
#             while len(new.tours) < m:
#                 new.tours.append([])
#             if len(new.tours) > m:
#                 new.tours = new.tours[:m]
#             # swap type
#             if random.random() < 0.5:
#                 idxs = [i for i in range(m) if len(new.tours[i]) > 1]
#                 if not idxs:
#                     continue
#                 r_idx = random.choice(idxs)
#                 p1 = random.randrange(len(new.tours[r_idx]))
#                 p2 = random.randrange(len(new.tours[r_idx]))
#                 candidate = intra_swap(new, r_idx, p1, p2)
#                 if candidate:
#                     new = candidate
#             else:
#                 non_empty = [i for i in range(m) if len(new.tours[i]) > 0]
#                 if len(non_empty) < 2:
#                     continue
#                 a, b = random.sample(non_empty, 2)
#                 p1 = random.randrange(len(new.tours[a]))
#                 p2 = random.randrange(len(new.tours[b]))
#                 candidate = inter_swap(new, a, p1, b, p2)
#                 if candidate:
#                     new = candidate

#     return new
def random_shake(sol: Solution, k: int, allow_dup_moves: bool = True) -> Solution:
    """
    Strong shaking that performs k randomized powerful moves.
    Moves may temporarily break multiplicities (Option A).
    - segment relocates (k_relocate)
    - 2-opt intra
    - 2-opt* cross
    - cross_exchange
    - relocate_sync_pair (if sync exists)
    - simple inter/intra insertions/swaps as fallback
    """
    new = sol.copy()
    inst = sol.instance
    m = inst.m
    _ensure_m_tours(new.tours, m)

    for _ in range(max(1, k)):
        r = random.random()

        # 20% big segment relocate
        if r < 0.20:
            # pick non-empty source
            non_empty = [i for i in range(m) if i < len(new.tours) and len(new.tours[i]) >= 2]
            if not non_empty:
                continue
            r_from = random.choice(non_empty)
            max_len = min(4, max(1, len(new.tours[r_from])//2))
            length = random.randint(1, max_len)
            start = random.randint(0, len(new.tours[r_from]) - length)
            r_to = random.randrange(m)
            pos = random.randrange(len(new.tours[r_to]) + 1)
            cand = k_relocate(new, r_from, start, length, r_to, pos)
            if cand:
                new = cand

        # 20% intra 2-opt
        elif r < 0.40:
            candidates = [i for i in range(m) if i < len(new.tours) and len(new.tours[i]) >= 4]
            if not candidates:
                continue
            r_idx = random.choice(candidates)
            L = len(new.tours[r_idx])
            i = random.randint(0, L-3)
            j = random.randint(i+2, L-1)
            cand = two_opt(new, r_idx, i, j)
            if cand:
                new = cand

        # 20% cross 2-opt*
        elif r < 0.60:
            candidates = [i for i in range(m) if i < len(new.tours) and len(new.tours[i]) >= 2]
            if len(candidates) < 2:
                continue
            r1, r2 = random.sample(candidates, 2)
            i = random.randint(1, len(new.tours[r1]) - 1) if len(new.tours[r1])>1 else 1
            j = random.randint(1, len(new.tours[r2]) - 1) if len(new.tours[r2])>1 else 1
            cand = two_opt_star(new, r1, i, r2, j)
            if cand:
                new = cand

        # 15% cross-exchange
        elif r < 0.75:
            idxs = [i for i in range(m) if i < len(new.tours) and len(new.tours[i]) >= 2]
            if len(idxs) < 2:
                continue
            r1, r2 = random.sample(idxs, 2)
            A = new.tours[r1]
            B = new.tours[r2]
            a1 = random.randint(0, len(A)-1)
            a2 = random.randint(a1+1, len(A))
            b1 = random.randint(0, len(B)-1)
            b2 = random.randint(b1+1, len(B))
            # limit segment sizes to small values
            if a2 - a1 > 4:
                a2 = a1 + random.randint(1,4)
                if a2 > len(A): a2 = len(A)
            if b2 - b1 > 4:
                b2 = b1 + random.randint(1,4)
                if b2 > len(B): b2 = len(B)
            if a1 < a2 and b1 < b2:
                cand = cross_exchange(new, r1, a1, a2, r2, b1, b2)
                if cand:
                    new = cand

        # 10% relocate sync pair
        elif r < 0.85:
            syncs = list(getattr(inst, "sync_customers", []))
            if syncs:
                cust = random.choice(syncs)
                cand = relocate_sync_pair(new, cust)
                if cand:
                    new = cand

        # fallback: random small inter/intra insertion/swap
        else:
            choice = random.random()
            if choice < 0.5:
                # insertion moves
                non_empty = [i for i in range(m) if i < len(new.tours) and len(new.tours[i]) > 0]
                if not non_empty:
                    continue
                a = random.choice(non_empty)
                if len(new.tours[a]) > 1 and random.random() < 0.5:
                    from_pos = random.randrange(len(new.tours[a]))
                    to_pos = random.randrange(len(new.tours[a]) + 1)
                    cand = intra_insertion(new, a, from_pos, to_pos)
                    if cand: new = cand
                else:
                    b = random.randrange(m)
                    if b == a or len(new.tours[a]) == 0:
                        continue
                    from_pos = random.randrange(len(new.tours[a]))
                    to_pos = random.randrange(len(new.tours[b]) + 1)
                    cand = inter_insertion(new, a, from_pos, b, to_pos)
                    if cand: new = cand
            else:
                # swap moves
                idxs = [i for i in range(m) if i < len(new.tours) and len(new.tours[i]) > 1]
                if idxs and random.random() < 0.5:
                    r_idx = random.choice(idxs)
                    p1 = random.randrange(len(new.tours[r_idx]))
                    p2 = random.randrange(len(new.tours[r_idx]))
                    cand = intra_swap(new, r_idx, p1, p2)
                    if cand: new = cand
                else:
                    non_empty = [i for i in range(m) if i < len(new.tours) and len(new.tours[i]) > 0]
                    if len(non_empty) < 2:
                        continue
                    a, b = random.sample(non_empty, 2)
                    p1 = random.randrange(len(new.tours[a]))
                    p2 = random.randrange(len(new.tours[b]))
                    cand = inter_swap(new, a, p1, b, p2)
                    if cand: new = cand

    return new