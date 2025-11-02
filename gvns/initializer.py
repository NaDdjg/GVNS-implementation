# gvns/initializer.py
import random
import math
from typing import List
from copy import deepcopy

from gvns.solution import Solution

def earliest_insertion(instance, tours, cust, vehicle_idx):
    """
    Try to insert cust into tours[vehicle_idx] at a position that minimizes
    increase in travel time while trying to respect time windows.
    Returns True if inserted, False otherwise.
    """
    TT = instance.travel_time
    depot = min(instance.nodes.keys())
    route = tours[vehicle_idx]
    best_pos = None
    best_cost = float('inf')
    # consider positions 0..len(route)
    for pos in range(len(route)+1):
        # estimate travel cost change: previous->cust + cust->next - prev->next
        prev = depot if pos == 0 else route[pos-1]
        nxt = depot if pos == len(route) else route[pos]
        delta = TT[prev][cust] + TT[cust][nxt] - TT[prev][nxt]
        if delta < best_cost:
            best_cost = delta
            best_pos = pos
    if best_pos is None:
        return False
    route.insert(best_pos, cust)
    return True

def random_insertion(instance, seed: int = 0):
    """
    Build an initial solution:
      - ensure each synchronized customer appears exactly sync_requirements[c] times
        and each occurrence is on a different vehicle
      - place remaining customers once each
    Strategy:
      1. Create empty tours for m vehicles
      2. For each sync customer, assign its two occurrences to two different vehicles:
         choose the two vehicles that currently have smallest number of nodes (balance)
         and try earliest insertion (greedy); if fails, force insert.
      3. For remaining customers, insert greedily into the vehicle/position with minimal cost
    Returns a Solution object (not evaluated).
    """
    random.seed(seed)
    m = instance.m
    tours = [[] for _ in range(m)]

    # 1) place synchronized customers (each required times on different vehicles)
    sync_list = list(instance.sync_customers)
    for sc in sync_list:
        req = instance.sync_requirements.get(sc, 2)
        # choose req distinct vehicles preferring least-loaded
        vehicle_indices = sorted(range(m), key=lambda x: len(tours[x]))
        selected = vehicle_indices[:req]
        for v in selected:
            inserted = earliest_insertion(instance, tours, sc, v)
            if not inserted:
                # fallback: append at end
                tours[v].append(sc)

    # 2) place non-synchronized customers once
    all_customers = [cid for cid in instance.nodes.keys() if cid != min(instance.nodes.keys())]
    # remove synchronized ones since they've been inserted
    remaining = [c for c in all_customers if c not in instance.sync_customers]

    # randomize remaining order to diversify initial solutions
    random.shuffle(remaining)
    for cust in remaining:
        # choose best vehicle to insert (min delta cost)
        best_v = None
        best_pos = None
        best_cost = float('inf')
        for v in range(m):
            # if inserting into a route that already contains this customer (shouldn't), skip
            if tours[v].count(cust) > 0:
                continue
            # try positions
            for pos in range(len(tours[v]) + 1):
                prev = min(instance.nodes.keys()) if pos == 0 else tours[v][pos-1]
                nxt = min(instance.nodes.keys()) if pos == len(tours[v]) else tours[v][pos]
                delta = instance.travel_time[prev][cust] + instance.travel_time[cust][nxt] - instance.travel_time[prev][nxt]
                if delta < best_cost:
                    best_cost = delta
                    best_v = v
                    best_pos = pos
        if best_v is None:
            # fallback: place in vehicle with smallest load
            v = min(range(m), key=lambda x: len(tours[x]))
            tours[v].append(cust)
        else:
            tours[best_v].insert(best_pos, cust)

    # final safety: ensure sync occurrences are on distinct vehicles
    # if any synchronized cust ended up twice on same vehicle, move a duplicate to another vehicle
    for sc in sync_list:
        occurrences = [(idx, pos) for idx, rt in enumerate(tours) for pos, val in enumerate(rt) if val == sc]
        if len(occurrences) <= 1:
            # if less than required, add missing occurrences on smallest routes
            needed = instance.sync_requirements.get(sc,2) - len(occurrences)
            for _ in range(needed):
                v = min(range(m), key=lambda x: len(tours[x]))
                tours[v].append(sc)
        else:
            # ensure occurrences are on different vehicles
            vehicles = [idx for idx, _ in occurrences]
            if len(set(vehicles)) < len(vehicles):
                # move duplicates to other vehicles
                for (idx, pos) in occurrences:
                    # if duplicate on same vehicle as another, move it
                    if vehicles.count(idx) > 1:
                        # find a vehicle without this sc
                        candidates = [v for v in range(m) if sc not in tours[v]]
                        if candidates:
                            newv = min(candidates, key=lambda x: len(tours[x]))
                            # remove one occurrence from idx
                            # find pos again (safely)
                            try:
                                tours[idx].remove(sc)
                            except ValueError:
                                continue
                            tours[newv].append(sc)

    sol = Solution(instance=instance, tours=tours)
    return sol
