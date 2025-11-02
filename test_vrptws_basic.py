import math
import random

from gvns.instance import Visit, Instance, build_travel_matrix
from gvns.solution import Solution, pretty_print_solution
from gvns.evaluate import evaluate_solution
from gvns.instance import paper_initial_fixed
from gvns.operators import shake_solution, clone_solution
from gvns.local_search import gvns

def build_tiny_instance():
    """
    Build a simple artificial VRPTW-S instance to test evaluation & initialization.
    Layout (Euclidean coordinates):

         (3) V3
          |
    (1) V1 ---- (2) V2          (4) V4
      \           \
       \           \
        \___________(0) Depot

    Sync pair: (V1,V2)
    """
    # Depot
    depot = Visit(id=0, x=0.0, y=0.0, service=0.0, tw_early=0.0, tw_late=1000.0)

    # Visits
    visits = {
        1: Visit(id=1, x=10.0, y=0.0, service=10.0, tw_early=0.0, tw_late=500.0),
        2: Visit(id=2, x=10.0, y=10.0, service=10.0, tw_early=0.0, tw_late=500.0),
        3: Visit(id=3, x=0.0, y=15.0, service=5.0, tw_early=0.0, tw_late=500.0),
        4: Visit(id=4, x=20.0, y=5.0, service=5.0, tw_early=0.0, tw_late=500.0),
    }

    # Create instance
    inst = Instance(
        name="tiny_test",
        depot=depot,
        visits=visits,
        horizon=600.0,
        sync_pairs=[(1, 2)],   # V1 and V2 must be synchronized
        sync_count=1,
    )

    build_travel_matrix(inst, speed=1.0)
    return inst


def main():
    inst = build_tiny_instance()
    sol, hist = gvns(inst, Tmax_seconds=100, contmax=10, p_init=1, p_max=4, verbose=True, seed=42, num_vehicles_estimate=2)
    print("Final history length:", len(hist))


if __name__ == "__main__":
    main()