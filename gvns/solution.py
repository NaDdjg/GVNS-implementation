from dataclasses import dataclass, field
from typing import List

BIG_M = 1_000_000.0

def safe_tt(TT, i, j):
    """safe access to travel-time matrix supporting list or dict form"""
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

    # utility
    def copy(self):
        new = Solution(
            instance=self.instance,
            tours=[list(t) for t in self.tours],
            w1=self.w1,
            w2=self.w2,
            alpha=self.alpha,
            M=self.M,
        )
        new.travel_time = self.travel_time
        new.penT = self.penT
        new.penH = self.penH
        new.penS = self.penS
        new.Fval = self.Fval
        new.PV = self.PV

        if hasattr(self, "_last_vehicle_visits"):
            new._last_vehicle_visits = [
                list(vehicle) for vehicle in self._last_vehicle_visits
            ]

        return new
    
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

        return self.w1, self.w2

    # core Evaluation
    def eval(self):
        """
        Canonical evaluation: run a single event-driven simulation that:
         - records per-vehicle visits (travel legs, arrival/start/depart)
         - uses those records to compute travel_total, penT, penH, penS and Fval
         - exactly mirrors the synchronization rules (required vehicles, eps alignment)
        """
        inst = self.instance
        TT = inst.travel_time
        nodes = inst.nodes
        m = inst.m
        horizon = inst.horizon
        sync_customers = getattr(inst, "sync_customers", set())
        sync_requirements = getattr(inst, "sync_requirements", {})
        depot_id = inst.depot_id 

        # prepare per-vehicle state
        tours = [list(t) for t in self.tours]  # copy
        if len(tours) < m:
            tours = tours + [[] for _ in range(m - len(tours))]
        elif len(tours) > m:
            tours = tours[:m]

        # assign back because later functions use self.tours!
        self.tours = tours
        
        vehicle_time = [0.0] * m
        vehicle_pos = [0] * m
        vehicle_done = [False] * m

        # initialize first arrival times (depot -> first node)
        for k in range(m):
            if k < len(tours) and len(tours[k]) > 0:
                vehicle_time[k] = safe_tt(TT, depot_id, tours[k][0])
            else:
                vehicle_done[k] = True

        # visit records: list of tuples
        # (node, travel, start, due, depart)
        vehicle_visits = [[] for _ in range(m)]
        sync_arrivals = {cid: [] for cid in sync_customers}
        # sync_served = set()

        # simulation loop (event-driven style similar to previous logic)
        progress = True
        iter_guard = 0
        max_iter = sum(len(t) for t in tours) * 10 + 500

        while progress and iter_guard < max_iter:
            progress = False
            iter_guard += 1

            for k in range(m):
                if vehicle_done[k]:
                    continue
                if vehicle_pos[k] >= len(tours[k]):
                    vehicle_done[k] = True
                    continue

                i = tours[k][vehicle_pos[k]]
                node_info = nodes[i]
                ready = node_info["ready"]
                due = node_info["due"]
                service = node_info["service"]

                # vehicle waits for ready if necessary
                if vehicle_time[k] < ready:
                    vehicle_time[k] = ready
                    progress = True

                # SYNC customer
                if i in sync_customers:
                    if k not in sync_arrivals[i]:
                        sync_arrivals[i].append(k)
                        progress = True

                    req = sync_requirements.get(i, 2)

                    # not enough vehicles yet, wait
                    if len(sync_arrivals[i]) < req:
                        continue

                    # we have required vehicles -> check alignment
                    arrival_times = [vehicle_time[v] for v in sync_arrivals[i]]
                    # synchronized -> start at latest arrival
                    start_time = max(arrival_times)

                    # record a visit for each arriving vehicle
                    for v in list(sync_arrivals[i]):
                        prev = depot_id if vehicle_pos[v] == 0 else tours[v][vehicle_pos[v] - 1]
                        travel_leg = TT[prev][i]
                        depart = start_time + service
                        vehicle_visits[v].append(
                            (i, travel_leg, start_time, due, depart)
                        )

                        # advance vehicle state
                        vehicle_pos[v] += 1
                        # set next arrival time = depart + travel(i,next) or return to depot
                        depart = start_time + service
                        if vehicle_pos[v] < len(tours[v]):
                            nxt = tours[v][vehicle_pos[v]]
                            vehicle_time[v] = depart + TT[i][nxt]
                        else:
                            vehicle_time[v] = depart + TT[i][depot_id]

                    # sync_served.add(i)
                    sync_arrivals[i] = []
                    progress = True

                # NON-SYNC customer
                else:
                    arrival = vehicle_time[k]
                    start = arrival if arrival >= ready else ready
                    depart = start + service

                    prev = depot_id if vehicle_pos[k] == 0 else tours[k][vehicle_pos[k] - 1]
                    travel_leg = TT[prev][i]

                    vehicle_visits[k].append(
                        (i, travel_leg, start, due, depart)
                    )

                    vehicle_pos[k] += 1
                    if vehicle_pos[k] < len(tours[k]):
                        nxt = tours[k][vehicle_pos[k]]
                        vehicle_time[k] = depart + TT[i][nxt]
                    else:
                        vehicle_time[k] = depart + TT[i][depot_id]
                    progress = True

        travel_total = 0.0
        penT = 0.0
        penH = 0.0

        for k in range(m):
            visits = vehicle_visits[k]
            if not visits:
                continue

            last_node = depot_id
            last_depart = 0.0

            for node, travel, start, due, depart in visits:
                travel_total += travel
                if start > due:
                    penT += (start - due)
                last_node = node
                last_depart = depart

            travel_total += TT[last_node][depot_id]

            end_time = last_depart + TT[last_node][depot_id]
            if end_time > horizon:
                penH += (end_time - horizon)

        # synchronization penalty for any sync customers that never got served
        n = 0
        for cid in nodes.keys():
            if cid == depot_id:
                continue
            # each non-sync customer requires 1 occurrence,
            # a sync customer requires sync_requirements[cid]
            n += sync_requirements.get(cid, 1)

        PV = sum(len(visits) for visits in vehicle_visits)
        self.PV = int(PV)
        
        penS = max(0, n - PV)

        # final objective
        fixed_sum = 0.0
        if hasattr(inst, 'fixed_cost'):
            for k in range(m):
                if k < len(tours) and len(tours[k]) > 0:
                    fixed_sum += float(inst.fixed_cost)

        self.travel_time = travel_total
        self.penT = penT
        self.penH = penH
        self.penS = penS
        self.Fval = (travel_total + fixed_sum) + self.w1 * penT + self.w2 * penH + self.M * penS

        # canonical timeline on the object for inspection
        self._last_vehicle_visits = vehicle_visits

        return self.Fval

    # logging 
    def compute_timeline_postsync(self):
        """
        Return the per-vehicle timeline produced by the canonical eval() run.
        If eval() wasn't called, run the canonical simulation once.
        """
        if not hasattr(self, "_last_vehicle_visits"):
            # run canonical eval (it will produce and store _last_vehicle_visits)
            self.eval()
        # return a results list in the same structure as before (route, visits, travel, end_time)
        inst = self.instance
        TT = inst.travel_time
        depot_id = min(inst.nodes.keys())
        results = []
        tours = [list(t) for t in self.tours]
        for k, visits in enumerate(self._last_vehicle_visits):
            travel_sum = 0.0
            last_depart = 0.0
            for rec in visits:
                travel_sum += float(rec.get("travel", 0.0))
                last_depart = float(rec.get("depart", last_depart))
            if visits:
                last_node = visits[-1]["node"]
                travel_sum += safe_tt(TT, last_node, depot_id)
                end_time = last_depart + safe_tt(TT, last_node, depot_id)
            else:
                end_time = 0.0
            results.append({"route": list(tours[k]) if k < len(tours) else [], "visits": visits, "travel": float(travel_sum), "end_time": float(end_time)})
        return results
