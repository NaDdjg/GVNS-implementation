from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING
import numpy as np
import random
import torch

if TYPE_CHECKING:           
    from .instance import Instance

@dataclass
class Solution:
    routes: List[List[int]]  
    f_value: Optional[float] = None
    penalties: Optional[Dict[str,float]] = None

def pretty_print_solution(sol: Solution):
    for r_idx, route in enumerate(sol.routes):
        print(f"Route {r_idx+1}: ", " -> ".join(map(str, route)))

def route_arrival_times(inst: "Instance", route: List[int], start_time=0.0):
    """
    compute arrival time at each visit and service start.
    assume the route starts at depot at start_time and ends at depot.
    returns list of tuples: (visit_id, arrival_time, start_service_time, departure_time)
    """
    res = []
    cur_time = start_time
    cur_node = inst.depot
    for v_id in route:
        travel = inst.travel_time[(cur_node.id, v_id)]
        arrival = cur_time + travel
        visit = inst.visits[v_id]
        start = max(arrival, visit.tw_early)  # wait allowed
        depart = start + visit.service
        res.append((v_id, arrival, start, depart))
        cur_time = depart
        cur_node = visit

    # travel back to depot
    return res

def set_deterministic(seed=42):
    """make runs reproducible for random and numpy. Call at start of each test/run."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    print(f"[deterministic] seeds set to {seed}")