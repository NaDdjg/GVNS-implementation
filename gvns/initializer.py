import random
import math
from typing import List
from copy import deepcopy

from gvns.solution import Solution

def route_prefix_time_estimate(instance, route, pos):
    """
    Approximate the vehicle arrival time at position `pos` in route (before visiting node at pos).
    We simulate up to pos (not including service at pos), using node ready/service and travel times.
    Returns estimated arrival time at the node that will be at index pos (i.e. time when vehicle reaches that node).
    If pos == len(route) it returns the arrival time at returning to depot after last node.
    """
    nodes = instance.nodes
    TT = instance.travel_time
    depot = min(nodes.keys())

    if not route:
        # arriving at first node: travel from depot
        return 0.0

    t = 0.0
    prev = depot
    for idx in range(min(pos, len(route))):
        cur = route[idx]
        # travel
        t += safe_tt(TT, prev, cur)
        # wait for ready if needed
        ready = nodes[cur]["ready"]
        if t < ready:
            t = ready
        # service
        t += nodes[cur]["service"]
        prev = cur

    # now t is depart time from prev; arrival at position 'pos' node is depart + travel(prev,posnode)
    if pos < len(route):
        nxt = route[pos]
        t += safe_tt(TT, prev, nxt)
    else:
        # arrival to depot if pos == len(route)
        t += safe_tt(TT, prev, depot)
    return t

def safe_tt(TT, i, j):
    try:
        if isinstance(TT, dict):
            return TT.get(i, {}).get(j, 0.0)
        else:
            return TT[i][j]
    except Exception:
        return 0.0

def earliest_insertion_time_aware(instance, tours, cust, vehicle_idx, lateness_weight=50.0, horizon_weight=50.0):
    """
    Insert cust into tours[vehicle_idx] at a position minimizing combined cost:
      travel_delta + lateness_weight * predicted_lateness + horizon_weight * predicted_horizon_over
    Returns True if inserted, False otherwise.
    """
    TT = instance.travel_time
    nodes = instance.nodes
    depot = min(nodes.keys())
    route = tours[vehicle_idx]
    best_pos = None
    best_cost = float('inf')

    for pos in range(len(route)+1):
        prev = depot if pos == 0 else route[pos-1]
        nxt = depot if pos == len(route) else route[pos]

        # travel delta
        delta_travel = safe_tt(TT, prev, cust) + safe_tt(TT, cust, nxt) - safe_tt(TT, prev, nxt)

        # estimate arrival/start at cust if inserted at pos
        # compute depart time from prev by simulating prefix
        if pos == 0:
            depart_prev = 0.0  # assume leaving depot at time 0
            # arrival at cust
            arr = depart_prev + safe_tt(TT, prev, cust)
        else:
            # simulate prefix up to prev
            arr_prev = route_prefix_time_estimate(instance, route, pos-1)
            # arr_prev is arrival at prev; depart_prev = max(arr_prev, ready_prev) + service_prev
            prev_ready = nodes[prev]["ready"]
            prev_service = nodes[prev]["service"]
            start_prev = max(arr_prev, prev_ready)
            depart_prev = start_prev + prev_service
            arr = depart_prev + safe_tt(TT, prev, cust)

        start = max(arr, nodes[cust]["ready"])
        lateness = max(0.0, start - nodes[cust]["due"])

        # approximate horizon overflow: estimate end_time if we insert here and then finish route
        # simple heuristic: assume the rest of route keeps same travel/service pattern,
        # compute depart at cust and then estimate time to return to depot after finishing route
        depart_after = start + nodes[cust]["service"]
        # compute time to finish the rest of route from position pos (we'll approximate by travel along remaining nodes)
        # quick approximate: add travel from cust->next + remaining travel of route (estimate by summing TT along current suffix)
        suffix_travel = 0.0
        prev_node = cust
        if pos < len(route):
            # travel cust -> route[pos], then add travel along suffix
            suffix_travel += safe_tt(TT, cust, route[pos])
            for a, b in zip(route[pos:], route[pos+1:]):
                suffix_travel += safe_tt(TT, a, b)
            # add travel last->depot
            last = route[-1]
            suffix_travel += safe_tt(TT, last, depot)
        else:
            # directly cust->depot
            suffix_travel += safe_tt(TT, cust, depot)

        est_end = depart_after + suffix_travel
        horizon_over = max(0.0, est_end - instance.horizon)

        cost = delta_travel + lateness_weight * lateness + horizon_weight * horizon_over

        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    if best_pos is None:
        return False
    tours[vehicle_idx].insert(best_pos, cust)
    return True

def random_insertion(instance, seed: int = 0):
    random.seed(seed)
    m = instance.m
    tours = [[] for _ in range(m)]

    # 1) improved place synchronized customers (try to place the pair jointly)
    sync_list = list(instance.sync_customers)
    for sc in sync_list:
        req = instance.sync_requirements.get(sc, 2)
        # if req != 2 we fallback to greedy single inserts
        if req == 2:
            # try candidate vehicle+pos combos (limit combinatorics)
            candidates = []
            for v in range(m):
                for pos in range(len(tours[v]) + 1):
                    est_arr = route_prefix_time_estimate(instance, tours[v], pos)
                    # arrival time at sc when inserted at pos on v
                    arr = est_arr
                    start = max(arr + safe_tt(instance.travel_time, min(instance.nodes.keys()) if pos==0 else (tours[v][pos-1]), sc), instance.nodes[sc]["ready"])
                    candidates.append((v, pos, start))
            # sort candidates by start time to find pairs that align
            candidates.sort(key=lambda x: x[2])
            placed = 0
            used = set()
            # try to find two distinct vehicles with close start times
            best_pair = None
            best_pair_cost = float('inf')
            # scan windows of candidate list to find close pairs on different vehicles
            for i in range(len(candidates)):
                for j in range(i+1, min(i+15, len(candidates))):  # look ahead a bit
                    v1, pos1, s1 = candidates[i]
                    v2, pos2, s2 = candidates[j]
                    if v1 == v2:
                        continue
                    # cost = abs(start diff) + travel deltas at those positions
                    prev1 = min(instance.nodes.keys()) if pos1 == 0 else tours[v1][pos1-1]
                    nxt1 = min(instance.nodes.keys()) if pos1 == len(tours[v1]) else tours[v1][pos1]
                    delta1 = safe_tt(instance.travel_time, prev1, sc) + safe_tt(instance.travel_time, sc, nxt1) - safe_tt(instance.travel_time, prev1, nxt1)

                    prev2 = min(instance.nodes.keys()) if pos2 == 0 else tours[v2][pos2-1]
                    nxt2 = min(instance.nodes.keys()) if pos2 == len(tours[v2]) else tours[v2][pos2]
                    delta2 = safe_tt(instance.travel_time, prev2, sc) + safe_tt(instance.travel_time, sc, nxt2) - safe_tt(instance.travel_time, prev2, nxt2)

                    pair_cost = abs(s1 - s2) + 0.5 * (delta1 + delta2)
                    if pair_cost < best_pair_cost:
                        best_pair_cost = pair_cost
                        best_pair = (v1, pos1, v2, pos2)
            if best_pair:
                v1, pos1, v2, pos2 = best_pair
                tours[v1].insert(pos1, sc)
                # adjust second pos if same vehicle index <-> insertion order issues
                if v2 == v1 and pos2 >= pos1:
                    pos2 += 1
                tours[v2].insert(pos2, sc)
            else:
                # fallback: simple balanced insertion using time-aware single insertion
                vehicle_indices = sorted(range(m), key=lambda x: len(tours[x]))
                for v in vehicle_indices[:req]:
                    earliest_insertion_time_aware(instance, tours, sc, v)
        else:
            # general approach for req != 2: do repeated single time-aware insertions on distinct vehicles
            vehicle_indices = sorted(range(m), key=lambda x: len(tours[x]))
            for v in vehicle_indices[:req]:
                earliest_insertion_time_aware(instance, tours, sc, v)

    # 2) place non-synchronized customers once (time-aware greedy)
    all_customers = [cid for cid in instance.nodes.keys() if cid != min(instance.nodes.keys())]
    remaining = [c for c in all_customers if c not in instance.sync_customers]
    random.shuffle(remaining)
    for cust in remaining:
        best_v = None
        best_pos = None
        best_cost = float('inf')
        for v in range(m):
            if tours[v].count(cust) > 0:
                continue
            for pos in range(len(tours[v]) + 1):
                prev = min(instance.nodes.keys()) if pos == 0 else tours[v][pos-1]
                nxt = min(instance.nodes.keys()) if pos == len(tours[v]) else tours[v][pos]
                delta = safe_tt(instance.travel_time, prev, cust) + safe_tt(instance.travel_time, cust, nxt) - safe_tt(instance.travel_time, prev, nxt)
                # estimate lateness at this insertion (quick)
                est_arr = route_prefix_time_estimate(instance, tours[v], pos)
                # arrival at cust = depart_prev + travel(prev,cust) (we approximated depart_prev in route_prefix_time_estimate)
                start = max(est_arr, instance.nodes[cust]["ready"])
                lateness = max(0.0, start - instance.nodes[cust]["due"])
                cost = delta + 50.0 * lateness
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_pos = pos
        if best_v is None:
            v = min(range(m), key=lambda x: len(tours[x]))
            tours[v].append(cust)
        else:
            tours[best_v].insert(best_pos, cust)

    # final safety: ensure sync occurrences on distinct vehicles (if duplicates remain, move them)
    for sc in sync_list:
        occurrences = [(idx, pos) for idx, rt in enumerate(tours) for pos, val in enumerate(rt) if val == sc]
        if len(occurrences) < instance.sync_requirements.get(sc,2):
            needed = instance.sync_requirements.get(sc,2) - len(occurrences)
            for _ in range(needed):
                v = min(range(m), key=lambda x: len(tours[x]))
                tours[v].append(sc)
        else:
            # move duplicates so each occurrence resides on a different vehicle
            vehicles = [idx for idx, _ in occurrences]
            if len(set(vehicles)) < len(vehicles):
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

    sol = Solution(instance=instance, tours=tours)
    return sol