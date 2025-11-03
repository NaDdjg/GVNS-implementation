# gvns/solution.py
from dataclasses import dataclass, field
from typing import List, Dict
import random
import math

BIG_M = 1_000_000.0

def safe_tt(TT, i, j):
    """Safe access to travel-time matrix supporting list or dict form."""
    try:
        if isinstance(TT, dict):
            return TT.get(i, {}).get(j, 0.0)
        else:
            return TT[i][j]
    except Exception:
        return 0.0


@dataclass
class Solution:
    instance: object
    tours: List[List[int]] = field(default_factory=list)
    w1: float = 0.5
    w2: float = 0.1
    alpha: float = 0.1
    M: float = BIG_M

    travel_time: float = 0.0
    penT: float = 0.0
    penH: float = 0.0
    penS: float = 0.0
    Fval: float = 0.0
    PV: int = 0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def copy(self):
        return Solution(
            instance=self.instance,
            tours=[list(t) for t in self.tours],
            w1=self.w1,
            w2=self.w2,
            alpha=self.alpha,
            M=self.M,
        )

    # ------------------------------------------------------------------
    # Core Evaluation
    # ------------------------------------------------------------------
    def eval(self):
        """
        Evaluate total cost including:
        - travel distance
        - lateness (PenT)
        - horizon (PenH)
        - synchronization (PenS)
        """
        self.repair()  # Ensure feasibility first

        inst = self.instance
        TT = inst.travel_time
        nodes = inst.nodes
        m = inst.m
        horizon = inst.horizon
        sync_customers = getattr(inst, "sync_customers", [])
        sync_requirements = getattr(inst, "sync_requirements", {})

        depot_id = min(nodes.keys())
        for nid, info in nodes.items():
            if info.get("ready", None) == 0 and info.get("service", None) == 0:
                depot_id = nid
                break

        vehicle_time = [0.0] * m
        vehicle_pos = [0] * m
        vehicle_done = [False] * m
        visited = {nid: 0 for nid in nodes if nid != depot_id}
        travel_total = 0.0
        penT = penH = penS = 0.0

        # Synchronization tracking
        sync_arrivals: Dict[int, List[int]] = {cid: [] for cid in sync_customers}
        sync_served = set()

        # Initialize first arrivals
        for k in range(m):
            if k < len(self.tours) and len(self.tours[k]) > 0:
                first = self.tours[k][0]
                vehicle_time[k] = safe_tt(TT, depot_id, first)
            else:
                vehicle_done[k] = True

        # Simulation loop
        progress = True
        iter_guard = 0
        max_iter = sum(len(t) for t in self.tours) * 10 + 200
        while progress and iter_guard < max_iter:
            progress = False
            iter_guard += 1

            for k in range(m):
                if vehicle_done[k]:
                    continue
                route = self.tours[k]
                if vehicle_pos[k] >= len(route):
                    vehicle_done[k] = True
                    continue

                i = route[vehicle_pos[k]]
                ready = nodes[i]["ready"]
                due = nodes[i]["due"]
                service = nodes[i]["service"]

                # Wait for ready time
                if vehicle_time[k] < ready:
                    vehicle_time[k] = ready

                # --- SYNCHRONIZED CUSTOMER ---
                if i in sync_customers:
                    arrivals = sync_arrivals[i]
                    if k not in arrivals:
                        arrivals.append(k)
                    req = sync_requirements.get(i, 2)

                    if len(arrivals) < req:
                        continue  # wait for others

                    # Check arrival alignment
                    arrival_times = [vehicle_time[v] for v in arrivals]
                    if max(arrival_times) - min(arrival_times) > 1e-3:
                        # Not synchronized properly → penalize
                        penS += 1.0
                        sync_arrivals[i] = []
                        continue

                    # Synchronize start
                    start_time = max(arrival_times)
                    if start_time > due:
                        penT += (start_time - due)

                    for v in arrivals:
                        visited[i] += 1
                        prev = depot_id if vehicle_pos[v] == 0 else self.tours[v][vehicle_pos[v] - 1]
                        travel_total += safe_tt(TT, prev, i)
                        vehicle_time[v] = start_time + service
                        vehicle_pos[v] += 1
                        if vehicle_pos[v] < len(self.tours[v]):
                            nxt = self.tours[v][vehicle_pos[v]]
                            vehicle_time[v] += safe_tt(TT, i, nxt)
                        else:
                            vehicle_time[v] += safe_tt(TT, i, depot_id)

                    sync_arrivals[i] = []
                    sync_served.add(i)

                    progress = True

                # --- NON-SYNCHRONIZED CUSTOMER ---
                else:
                    if vehicle_time[k] > due:
                        penT += (vehicle_time[k] - due)
                    prev = depot_id if vehicle_pos[k] == 0 else route[vehicle_pos[k] - 1]
                    travel_total += safe_tt(TT, prev, i)
                    visited[i] += 1
                    vehicle_time[k] += service
                    vehicle_pos[k] += 1
                    if vehicle_pos[k] < len(route):
                        nxt = route[vehicle_pos[k]]
                        vehicle_time[k] += safe_tt(TT, i, nxt)
                    else:
                        vehicle_time[k] += safe_tt(TT, i, depot_id)
                    progress = True

        # Horizon penalties
        for k in range(m):
            if vehicle_time[k] > horizon:
                penH += (vehicle_time[k] - horizon)

        # Synchronization penalty
        for cid in sync_customers:
            if cid not in sync_served:
                penS += 1.0

        # Final objective
        self.travel_time = travel_total
        self.penT = penT
        self.penH = penH
        self.penS = penS
        self.PV = sum(v > 0 for v in visited.values())
        self.Fval = travel_total + self.w1 * penT + self.w2 * penH + self.M * penS
        return self.Fval

    # ------------------------------------------------------------------
    # Repair and penalty adaptation
    # ------------------------------------------------------------------
    def repair(self):
        """
        Ensures each customer appears the correct number of times:
        - normal customers: 1 occurrence
        - synchronized customers: required # occurrences (usually 2)
        - removes extras, reinserts missing ones randomly
        """
        inst = self.instance
        sync_customers = getattr(inst, "sync_customers", [])
        sync_req = getattr(inst, "sync_requirements", {})
        depot_id = min(inst.nodes.keys())
        non_depot_nodes = [nid for nid in inst.nodes.keys() if nid != depot_id]

        from collections import Counter
        counter = Counter()
        for rt in self.tours:
            for v in rt:
                counter[v] += 1

        # Remove duplicates beyond allowed count
        for node, count in list(counter.items()):
            allowed = sync_req.get(node, 2 if node in sync_customers else 1)
            if count > allowed:
                excess = count - allowed
                for rt in self.tours:
                    while node in rt and excess > 0:
                        rt.remove(node)
                        excess -= 1

        # Insert missing customers
        for node in non_depot_nodes:
            allowed = sync_req.get(node, 2 if node in sync_customers else 1)
            current = sum(rt.count(node) for rt in self.tours)
            if current < allowed:
                for _ in range(allowed - current):
                    chosen_route = random.randint(0, len(self.tours) - 1)
                    pos = random.randint(0, len(self.tours[chosen_route]))
                    self.tours[chosen_route].insert(pos, node)

        return self

    def self_adjust_penalties(self):
        """
        dynamically adjust penalty weights 
        """
        if self.penT > 0:
            self.w1 *= (1.0 + self.alpha)
        else:
            self.w1 *= (1.0 - self.alpha)
        if self.penH > 0:
            self.w2 *= (1.0 + self.alpha)
        else:
            self.w2 *= (1.0 - self.alpha)

        # self.w1 = max(1e-6, min(self.w1, 1e6))
        # self.w2 = max(1e-6, min(self.w2, 1e6))
        self.eval()
        return self.w1, self.w2