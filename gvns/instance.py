from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import math, random, os
from .utils import pick_vehicle_avoiding_partners

SEED = 42

@dataclass
class Visit:
    id: int
    x: float
    y: float
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
    sync_count: int = 0

def euclid_time(a: Visit, b: Visit, speed=1.0):
    dist = math.hypot(a.x - b.x, a.y - b.y)
    return dist / speed

def build_travel_matrix(inst: Instance, speed=1.0):
    # build from depot + all visits:
    all_nodes = [inst.depot] + [inst.visits[i] for i in sorted(inst.visits.keys())]
    T = {}
    for i, vi in enumerate(all_nodes):
        for j, vj in enumerate(all_nodes):
            T[(vi.id, vj.id)] = euclid_time(vi, vj, speed)
    inst.travel_time = T

# parser with synch visits 
def parse_vrptws_txt(path):
    visits = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # --- Parse metadata ---
    try:
        num_vehicles = int(lines[0])
        num_customers = int(lines[1])
        horizon = float(lines[2])
    except Exception as e:
        print(f"[parse_vrptws_txt] Metadata parsing failed for {path}: {e}")
        return None

    # Optional sync info (not always present)
    sync_count = 0
    sync_ids = []
    if len(lines) > 3:
        try:
            sync_count = int(lines[3])
        except:
            sync_count = 0
    if sync_count > 0 and len(lines) > 4:
        sync_ids = [int(x) for x in lines[4].split()]

    # --- Parse visits ---
    data_start_idx = 5 if sync_count > 0 else 3
    for l in lines[data_start_idx:]:
        parts = l.strip().split()
        if len(parts) < 6:
            continue
        vid = int(parts[0])
        x, y = float(parts[1]), float(parts[2])
        ready, due = float(parts[3]), float(parts[4])
        service = float(parts[5])

        visits[vid] = Visit(
            id=vid, x=x, y=y,
            service=service, tw_early=ready, tw_late=due,
            original_id=vid
        )

    # Depot
    dep_id = 0 if 0 in visits else min(visits.keys())
    depot_due = visits[dep_id].tw_late
    depot = visits.pop(dep_id, None)
    if depot is None:
        print(f"[parse_vrptws_txt] No depot found in {path}")
        return None

    inst = Instance(name=os.path.basename(path), depot=depot)
    inst.visits = visits
    inst.horizon = depot_due
    inst.sync_count = sync_count

    # -Synchronized visits
    inst.sync_pairs = []
    if sync_count > 0 and sync_ids:
        print(f"[parse_vrptws_txt] Duplicating {len(sync_ids)} synchronized visits...")
        next_id = max(visits.keys()) + 1
        for sid in sync_ids:
            if sid not in visits:
                print(f"  [warning] Sync id {sid} not found among visits, skipping.")
                continue

            orig = visits[sid]
            newv = Visit(
                id=next_id,
                x=orig.x,
                y=orig.y,
                service=orig.service,
                tw_early=orig.tw_early,
                tw_late=orig.tw_late,
                original_id=sid
            )
            visits[next_id] = newv
            inst.sync_pairs.append((sid, next_id))
            next_id += 1
        print(f"  [sync] Created {len(inst.sync_pairs)} duplicate pairs: {inst.sync_pairs}")

    build_travel_matrix(inst)
    # logs
    print(f"[parse_vrptws_txt] Parsed {path}: {len(visits)} visits, "
            f"sync_count={sync_count}, sync_pairs={inst.sync_pairs}", 
            f"horizon={inst.horizon}")

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
        # attempt to pick a vehicle that does not already hold the partner(s)
        if avoid_pair and cid in sync_map:
            partners = list(sync_map[cid])
            # pick a vehicle uniform randomly but avoid vehicles that already have any partner
            candidate_vehicles = list(range(m))
            random.shuffle(candidate_vehicles)
            chosen = None
            for j in candidate_vehicles:
                bad = False
                for p in partners:
                    if VF.get(p, -1) == j:
                        bad = True
                        break
                if not bad:
                    chosen = j
                    break
            if chosen is None:
                # fallback: random
                chosen = random.randrange(m)
        else:
            chosen = random.randrange(m)

        VF[cid] = chosen
        routes[chosen].append(cid)

    sol = clone_solution(Solution(routes=[list(r) for r in routes]))
    eval_sol = evaluate_solution(inst, sol, 0.5, 0.1, 1e6)
    return sol, eval_sol

# ---- only for tests without benchmark 
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
#----