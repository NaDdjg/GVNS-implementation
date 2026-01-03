import random
from typing import List, Optional
from gvns.solution import Solution

# ============================================================
# utilities
# ============================================================
def safe_tt(TT, i, j):
    try:
        if isinstance(TT, dict):
            return TT.get(i, {}).get(j, 0.0)
        else:
            return TT[i][j]
    except Exception:
        return 0.0

def compute_route_prefixes(instance, route):
    """
    compute arrival/start/depart arrays for a route (depart depot at time 0).
    returns (arrivals, starts, departs, return_to_depot_time).
    """
    nodes = instance.nodes
    TT = instance.travel_time
    depot = min(nodes.keys())

    L = len(route)
    arrivals = [0.0] * L
    starts = [0.0] * L
    departs = [0.0] * L

    t = 0.0
    prev = depot
    for idx in range(L):
        cur = route[idx]
        t += safe_tt(TT, prev, cur)
        arrivals[idx] = t
        start = max(t, nodes[cur]["ready"])
        starts[idx] = start
        depart = start + nodes[cur]["service"]
        departs[idx] = depart
        t = depart
        prev = cur

    if L == 0:
        return arrivals, starts, departs, 0.0
    last = route[-1]
    return_to_depot = t + safe_tt(TT, last, depot)
    return arrivals, starts, departs, return_to_depot

def evaluate_insertion_cost(instance, route, prefixes, pos, cust,
                            lateness_weight=10.0, horizon_weight=10.0):
    """
    quick cost estimate for inserting cust at position pos in route.
    returns (cost, start_at_cust, lateness, est_end_time).
    """
    nodes = instance.nodes
    TT = instance.travel_time
    depot = min(nodes.keys())

    arrivals, starts, departs, return_to_depot = prefixes
    L = len(route)

    prev = depot if pos == 0 else route[pos-1]
    nxt = depot if pos == L else route[pos]

    delta_travel = safe_tt(TT, prev, cust) + safe_tt(TT, cust, nxt) - safe_tt(TT, prev, nxt)

    if pos == 0:
        depart_prev = 0.0
        arr = depart_prev + safe_tt(TT, prev, cust)
    else:
        arr = departs[pos-1] + safe_tt(TT, prev, cust)

    start = max(arr, nodes[cust]["ready"])
    lateness = max(0.0, start - nodes[cust]["due"])

    # quick suffix travel estimate
    suffix_travel = 0.0
    if pos < L:
        suffix_travel += safe_tt(TT, cust, route[pos])
        for a, b in zip(route[pos:], route[pos+1:]):
            suffix_travel += safe_tt(TT, a, b)
        suffix_travel += safe_tt(TT, route[-1], depot)
    else:
        suffix_travel += safe_tt(TT, cust, depot)

    depart_after = start + nodes[cust]["service"]
    est_end = depart_after + suffix_travel
    horizon_over = max(0.0, est_end - instance.horizon)

    cost = delta_travel + lateness_weight * lateness + horizon_weight * horizon_over
    return cost, start, lateness, est_end

def best_insertion_for_vehicle(instance, tours, cust, v, prefixes=None,
                               sample_pos_limit=12, lateness_weight=10.0, horizon_weight=10.0):
    """
    return best insertion info (pos, cost, start, lateness, est_end) for customer cust on vehicle v.
    for long routes it samples candidate positions to keep runtime bounded.
    """
    route = tours[v]
    L = len(route)
    if prefixes is None:
        prefixes = compute_route_prefixes(instance, route)

    # decide positions to evaluate
    if L <= sample_pos_limit:
        positions = list(range(L+1))
    else:
        positions = set()
        # start area
        positions.update(range(min(3, L+1)))
        # end area
        positions.update(range(max(0, L-2), L+1))
        # middle evenly spaced
        mid_slots = max(0, sample_pos_limit - len(positions))
        if mid_slots > 0:
            step = max(1, L // (mid_slots + 1))
            for i in range(step, L, step):
                positions.add(i)
        positions = sorted([p for p in positions if 0 <= p <= L])

    best = None
    best_info = None
    for pos in positions:
        cost, start, lateness, est_end = evaluate_insertion_cost(
            instance, route, prefixes, pos, cust,
            lateness_weight=lateness_weight, horizon_weight=horizon_weight
        )
        if best is None or cost < best:
            best = cost
            best_info = (pos, cost, start, lateness, est_end)
    return best_info

# ============================================================
# main initializer
# ============================================================
def random_insertion(instance, seed: int = 0,
                     lateness_weight: float = 10.0,
                     horizon_weight: float = 10.0,
                     sample_pos_limit: int = 14,
                     top_k_vehicles_for_sync: int = 6):
    """
    build an initial solution:
      - balanced assignment across vehicles (avoids dumping all customers on few vehicles)
      - sync customers placed on distinct, least-loaded vehicles
    returns a Solution(instance, tours)
    """
    random.seed(seed)
    m = instance.m
    nodes = instance.nodes
    depot = min(nodes.keys())

    # prepare empty tours
    tours: List[List[int]] = [[] for _ in range(m)]

    total_customers = len([nid for nid in nodes.keys() if nid != depot])
    # target balancing parameters
    target_load = max(1, total_customers // m)
    max_load = max(2, target_load + 2)  # allow small slack

    # helper: candidate vehicles ordered by current load (least loaded first)
    def candidate_vehicles(limit: Optional[int] = None):
        order = sorted(range(m), key=lambda x: (len(tours[x]), x))
        return order if limit is None else order[:limit]

    sync_list = list(instance.sync_customers)

    # place synchronized customers (one-by-one but ensure distinct vehicles & balance)
    for sc in sync_list:
        req = instance.sync_requirements.get(sc, 2)
        if req <= 0:
            continue

        # choose vehicles greedily among least-loaded ones
        chosen = []
        for _ in range(req):
            # consider only vehicles under max_load, otherwise consider full list
            cand_vs = [v for v in candidate_vehicles(limit=m) if len(tours[v]) < max_load and v not in chosen]
            if not cand_vs:
                cand_vs = [v for v in candidate_vehicles(limit=m) if v not in chosen]
            # evaluate best insertion among top candidates (limit to small number for speed)
            cand_vs = cand_vs[:max(6, top_k_vehicles_for_sync)]
            best_choice = None
            best_score = None
            for v in cand_vs:
                prefixes = compute_route_prefixes(instance, tours[v])
                info = best_insertion_for_vehicle(instance, tours, sc, v, prefixes=prefixes,
                                                  sample_pos_limit=sample_pos_limit,
                                                  lateness_weight=lateness_weight, horizon_weight=horizon_weight)
                if info is None:
                    continue
                pos, cost, start, lateness, est_end = info
                # score
                # prefer balanced (few nodes) and low lateness/align
                score = cost + 0.01 * len(tours[v])
                if best_score is None or score < best_score:
                    best_score = score
                    best_choice = (v, pos)
            if best_choice is None:
                # fallback: smallest route append
                v = min(range(m), key=lambda x: len(tours[x]))
                tours[v].append(sc)
                chosen.append(v)
            else:
                v, pos = best_choice
                tours[v].insert(pos, sc)
                chosen.append(v)

    # place remaining customers (balanced, sample positions)
    all_customers = [cid for cid in nodes.keys() if cid != depot]
    remaining = [c for c in all_customers if c not in instance.sync_customers]
    random.shuffle(remaining)

    # maintain prefixes cache per vehicle to avoid repeated recompute
    prefixes_cache = [compute_route_prefixes(instance, tours[v]) for v in range(m)]

    for cust in remaining:
        # consider a small set of candidate vehicles (least loaded)
        cand_vs = candidate_vehicles(limit=min(m, max(6, m // 4)))
        best_global = None  # (score, v, pos, start, lateness, est_end)
        for v in cand_vs:
            prefixes = prefixes_cache[v]
            info = best_insertion_for_vehicle(instance, tours, cust, v, prefixes=prefixes,
                                              sample_pos_limit=sample_pos_limit,
                                              lateness_weight=lateness_weight, horizon_weight=horizon_weight)
            if info is None:
                continue
            pos, cost, start, lateness, est_end = info
            # small tie-break: prefer less loaded vehicles
            score = cost + 0.01 * len(tours[v])
            if best_global is None or score < best_global[0]:
                best_global = (score, v, pos, start, lateness, est_end)

        if best_global is None:
            # append on least loaded
            v = min(range(m), key=lambda x: len(tours[x]))
            tours[v].append(cust)
            prefixes_cache[v] = compute_route_prefixes(instance, tours[v])
        else:
            _, v, pos, _, _, _ = best_global
            tours[v].insert(pos, cust)
            prefixes_cache[v] = compute_route_prefixes(instance, tours[v])

    # final safety & balancing pass
    # ensure each sync customer occurrences are on distinct vehicles and required count met
    for sc in sync_list:
        occurrences = [(idx, pos) for idx, rt in enumerate(tours) for pos, val in enumerate(rt) if val == sc]
        needed = instance.sync_requirements.get(sc, 2)
        if len(occurrences) < needed:
            for _ in range(needed - len(occurrences)):
                v = min(range(m), key=lambda x: len(tours[x]))
                tours[v].append(sc)
        else:
            vehicles = [idx for idx, _ in occurrences]
            if len(set(vehicles)) < len(vehicles):
                # move duplicates to least-loaded vehicles without sc
                for idx, pos in occurrences:
                    if vehicles.count(idx) > 1:
                        candidates = [v for v in range(m) if sc not in tours[v]]
                        if candidates:
                            newv = min(candidates, key=lambda x: len(tours[x]))
                            try:
                                tours[idx].remove(sc)
                            except ValueError:
                                continue
                            tours[newv].append(sc)

    # final balancing fix
    # ensure no route is enormously larger than average
    avg = max(1, total_customers // max(1, m))
    max_allowed = max(avg + 3, max_load)
    for v in range(m):
        while len(tours[v]) > max_allowed:
            # move one customer from heavy route to lightest route
            light_v = min(range(m), key=lambda x: len(tours[x]))
            if light_v == v:
                break
            # pick a customer that is not a sync placed duplicated or that won't break sync requirement
            moved = None
            for pos, cust in enumerate(tours[v]):
                # avoid moving a sync occurrence if it would cause duplicates on same vehicle (try to keep them distinct)
                if cust in instance.sync_customers:
                    # ensure there is another occurrence elsewhere to keep requirement
                    occs = sum(rt.count(cust) for rt in tours)
                    if occs <= instance.sync_requirements.get(cust, 1):
                        continue
                moved = cust
                break
            if moved is None:
                break
            try:
                tours[v].remove(moved)
                tours[light_v].append(moved)
            except ValueError:
                break

    sol = Solution(instance=instance, tours=tours)
    return sol