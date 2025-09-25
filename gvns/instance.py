from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import math, random, os
from .solution import Solution
from .evaluate import is_feasible, quick_repair_greedy
from .operators import clone_solution

SEED = 42

@dataclass
class Visit:
    id: int
    x: float
    y: float
    demand: float = 0.0
    service: float = 0.0
    tw_early: float = 0.0
    tw_late: float = 1e9

    # for synchronized copies, keep original_id link
    original_id: Optional[int] = None

@dataclass
class Instance:
    name: str
    depot: Visit
    visits: Dict[int, Visit] = field(default_factory=dict)  # id = visit
    travel_time: Dict[Tuple[int,int], float] = field(default_factory=dict)
    horizon: float = 10000.0
    sync_pairs: List[Tuple[int,int]] = field(default_factory=list) 

def euclid_time(a: Visit, b: Visit, speed=1.0):
    dist = math.hypot(a.x - b.x, a.y - b.y)
    return dist / speed

def build_travel_matrix(inst: Instance, speed=1.0):
    ids = [0] + sorted(inst.visits.keys()) + [max(inst.visits.keys())+1]
    # instead build from depot + all visits:
    all_nodes = [inst.depot] + [inst.visits[i] for i in sorted(inst.visits.keys())]
    T = {}
    for i, vi in enumerate(all_nodes):
        for j, vj in enumerate(all_nodes):
            T[(vi.id, vj.id)] = euclid_time(vi, vj, speed)
    inst.travel_time = T

# # random insertion initial solution generator
# def random_insertion(inst: Instance, num_vehicles_estimate=5, seed=SEED):
#     """
#     simple random insertion:
#     - start with empty routes number = num_vehicles_estimate
#     - shuffle visits (excluding depot)
#     - insert customers one by one: choose a random vehicle that does not violate the "same vehicle" rule for sync pairs
#     - if no vehicle can accept, create a new one
#     (i'll keep it like that for now but can be improved with greedy insertion)
#     """
#     random.seed(seed)
#     unassigned = set(inst.visits.keys())
#     routes = [[] for _ in range(num_vehicles_estimate)]
#     # map visit -> assigned route index
#     assigned = {}
#     def violates_same_route(candidate, route_idx):
#         # candidate is id , will check if any synced partner of candidate is already in route route_idx
#         for (a,b) in inst.sync_pairs:
#             if candidate == a and b in routes[route_idx]:
#                 return True
#             if candidate == b and a in routes[route_idx]:
#                 return True
#         return False

#     visit_list = list(inst.visits.keys())
#     random.shuffle(visit_list)
#     for v in visit_list:
#         placed = False
#         vehicle_indices = list(range(len(routes)))
#         random.shuffle(vehicle_indices)
#         for ridx in vehicle_indices:
#             if not violates_same_route(v, ridx):
#                 routes[ridx].append(v)
#                 assigned[v] = ridx
#                 placed = True
#                 break
#         if not placed:
#             # create new route
#             routes.append([v])
#             assigned[v] = len(routes)-1
#     # prune empty routes
#     routes = [r for r in routes if len(r)>0]
#     return Solution(routes=routes)

def parse_vrptws_txt(path):
    visits = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]  # remove empty lines

    # skip first 5 rows = metadata
    data_lines = lines[5:]

    for l in data_lines:
        parts = l.strip().split()
        if len(parts) < 7:
            continue
        try:
            vid = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            demand = float(parts[3])
            ready, due = float(parts[4]), float(parts[5])
            service = float(parts[6])
        except ValueError:
            continue

        visits[vid] = Visit(
            id=vid, x=x, y=y, demand=demand,
            service=service, tw_early=ready, tw_late=due,
            original_id=vid
        )

    dep_id = 0 if 0 in visits else min(visits.keys())
    depot = visits.pop(dep_id)

    inst = Instance(name=os.path.basename(path), depot=depot)
    inst.visits = visits
    build_travel_matrix(inst)
    return inst


def generate_synthetic_vrptw(n_customers=10, tw_window=50, service_mean=10.0):
    """
    create a small random VRPTW instance for testing.
    Depot coordinates at (50,50), customers in 0..100 square.
    """
    depot = Visit(id=0, x=50.0, y=50.0, service=0.0, tw_early=0.0, tw_late=1e9)
    inst = Instance(name=f"toy_{n_customers}", depot=depot)
    for i in range(1, n_customers+1):
        x = random.uniform(0,100)
        y = random.uniform(0,100)
        service = max(1.0, random.gauss(service_mean, service_mean*0.2))
        earliest = random.uniform(0, tw_window*0.5)
        latest = earliest + random.uniform(10, tw_window)
        inst.visits[i] = Visit(id=i, x=x, y=y, service=service, tw_early=earliest, tw_late=latest, original_id=i)
    build_travel_matrix(inst)
    inst.horizon = 8*60  
    return inst

def augment_with_synchronization(inst: Instance, sync_fraction=0.2, seed=SEED):
    """
    create synchronized duplicates of some customers.
    for each selected customer, create a new Visit with new id and identical coords/time windows,
    and add the pair (orig_id, new_id) to inst.sync_pairs.
    """
    random.seed(seed)
    n = len(inst.visits)
    k = max(1, int(round(n * sync_fraction)))
    candidates = list(inst.visits.keys())
    random.shuffle(candidates)
    selected = candidates[:k]
    next_id = max(inst.visits.keys()) + 1
    for orig in selected:
        v = inst.visits[orig]
        newv = Visit(id=next_id, x=v.x, y=v.y, service=v.service,
                     tw_early=v.tw_early, tw_late=v.tw_late, original_id=orig)
        inst.visits[next_id] = newv
        inst.sync_pairs.append((orig, next_id))
        next_id += 1
    build_travel_matrix(inst)
    return inst

def paper_initial_fixed(inst, num_vehicles_estimate: int = 30, seed: int = None, avoid_pair: bool = True):
    """
      - for each customer choose a vehicle j = rand()%m
      - if avoid_pair=True: while partner is already assigned to j, choose another j 
      - set VF[cid] = j
    it returns (Solution, eval).
    """
    from .solution import Solution
    from .operators import clone_solution
    from .evaluate import evaluate_solution

    if seed is not None:
        random.seed(seed)

    customer_ids = sorted(list(inst.visits.keys()))
    m = max(1, num_vehicles_estimate)
    VF = {cid: -1 for cid in customer_ids}
    routes = [[] for _ in range(m)]
    sync_map = {}
    for (a,b) in getattr(inst, "sync_pairs", []) or []:
        sync_map.setdefault(a, set()).add(b)
        sync_map.setdefault(b, set()).add(a)

    for cid in customer_ids:
        j = random.randrange(m)
        if avoid_pair:
            partner = None
            partners = sync_map.get(cid, None)
            if partners:
                partner = next(iter(partners))
            # while partner already assigned to j, pick another
            tries = 0
            while partner is not None and VF.get(partner, -1) == j and tries < 5*m:
                j = random.randrange(m)
                tries += 1
        VF[cid] = j
        routes[j].append(cid)

    sol = clone_solution(Solution(routes=[list(r) for r in routes]))
    eval_sol = evaluate_solution(inst, sol, 0.5, 0.1, None)
    return sol, eval_sol