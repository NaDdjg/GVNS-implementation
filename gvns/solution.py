from dataclasses import dataclass, field
from typing import List, Dict
import random
import math

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
                [dict(rec) for rec in vehicle] for vehicle in self._last_vehicle_visits
            ]

        return new

    # core Evaluation
    def eval(self):
        """
        Canonical evaluation: run a single event-driven simulation that:
         - records per-vehicle visits (travel legs, arrival/start/depart)
         - uses those records to compute travel_total, penT, penH, penS and Fval
         - exactly mirrors the synchronization rules (required vehicles, eps alignment)
        """
        # enforce occurrence counts first
        self.repair()

        inst = self.instance
        TT = inst.travel_time
        nodes = inst.nodes
        m = inst.m
        horizon = inst.horizon
        sync_customers = getattr(inst, "sync_customers", set())
        sync_requirements = getattr(inst, "sync_requirements", {})

        # detect depot id (same logic you already used)
        depot_id = min(nodes.keys())
        for nid, info in nodes.items():
            if info.get("ready", None) == 0 and info.get("service", None) == 0:
                depot_id = nid
                break

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

        # storage for recorded visits (the canonical timeline)
        vehicle_visits = [[] for _ in range(m)]
        sync_arrivals = {cid: [] for cid in sync_customers}
        sync_served = set()

        # simulation loop (event-driven style similar to previous logic)
        progress = True
        iter_guard = 0
        max_iter = sum(len(t) for t in tours) * 10 + 500
        eps_sync = 1e-3

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

                # SYNC customer
                if i in sync_customers:
                    if k not in sync_arrivals[i]:
                        sync_arrivals[i].append(k)
                    req = sync_requirements.get(i, 2)

                    # not enough vehicles yet, wait
                    if len(sync_arrivals[i]) < req:
                        continue

                    # we have required vehicles -> check alignment
                    arrival_times = [vehicle_time[v] for v in sync_arrivals[i]]
                    if max(arrival_times) - min(arrival_times) > eps_sync:
                        # desynchronization -> penalize and drop buffered arrivals (per paper)
                        # note we only increment penS after timeline creation (see below),
                        # but for bookkeeping we may mark an attempted sync failure here:
                        # reset arrival buffer (they will try again later)
                        sync_arrivals[i] = []
                        continue

                    # synchronized -> start at latest arrival
                    start_time = max(arrival_times)

                    # record a visit for each arriving vehicle
                    for v in list(sync_arrivals[i]):
                        prev = depot_id if vehicle_pos[v] == 0 else tours[v][vehicle_pos[v] - 1]
                        travel_leg = safe_tt(TT, prev, i)
                        # record same fields as compute_timeline_postsync
                        vehicle_visits[v].append({
                            "node": i,
                            "prev": prev,
                            "travel": float(travel_leg),
                            "arrival": float(vehicle_time[v]),
                            "start": float(start_time),
                            "service": float(service),
                            "depart": float(start_time + service),
                            "due": float(due)
                        })
                        # advance vehicle state
                        vehicle_pos[v] += 1
                        # set next arrival time = depart + travel(i,next) or return to depot
                        depart = start_time + service
                        if vehicle_pos[v] < len(tours[v]):
                            nxt = tours[v][vehicle_pos[v]]
                            vehicle_time[v] = depart + safe_tt(TT, i, nxt)
                        else:
                            vehicle_time[v] = depart + safe_tt(TT, i, depot_id)

                    sync_served.add(i)
                    sync_arrivals[i] = []
                    progress = True

                # NON-SYNC customer
                else:
                    arrival = vehicle_time[k]
                    start = arrival if arrival >= ready else ready
                    depart = start + service

                    prev = depot_id if vehicle_pos[k] == 0 else tours[k][vehicle_pos[k] - 1]
                    travel_leg = safe_tt(TT, prev, i)
                    vehicle_visits[k].append({
                        "node": i,
                        "prev": prev,
                        "travel": float(travel_leg),
                        "arrival": float(arrival),
                        "start": float(start),
                        "service": float(service),
                        "depart": float(depart),
                        "due": float(due)
                    })

                    vehicle_pos[k] += 1
                    if vehicle_pos[k] < len(tours[k]):
                        nxt = tours[k][vehicle_pos[k]]
                        vehicle_time[k] = depart + safe_tt(TT, i, nxt)
                    else:
                        vehicle_time[k] = depart + safe_tt(TT, i, depot_id)
                    progress = True

        # Now compute global aggregates *from the recorded timeline* (single canonical source)
        travel_total = 0.0
        for k in range(m):
            visits = vehicle_visits[k]
            for rec in visits:
                travel_total += float(rec.get("travel", 0.0))
            if visits:
                # add return to depot
                last_node = visits[-1]["node"]
                travel_total += safe_tt(TT, last_node, depot_id)

        # compute penT from recorded visits: lateness = max(0, start - due)
        penT = 0.0
        for k in range(m):
            for rec in vehicle_visits[k]:
                late = max(0.0, rec["start"] - rec["due"])
                penT += late

        # compute penH (horizon) from end_time per vehicle
        penH = 0.0
        for k in range(m):
            visits = vehicle_visits[k]
            if visits:
                last_depart = float(visits[-1]["depart"])
                last_node = visits[-1]["node"]
                end_time = last_depart + safe_tt(TT, last_node, depot_id)
                if end_time > horizon:
                    penH += (end_time - horizon)

        # synchronization penalty for any sync customers that never got served
        # penS = 0.0
        # for cid in sync_customers:
        #     if cid not in sync_served:
        #         penS += 1.0
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

    # repair and penalty adaptation
    def repair(self):
        """
        ensures each customer appears the correct number of times:
        - normal customers: 1 occurrence
        - synchronized customers: required # occurrences (will be 2 in our case)
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

        # remove duplicates beyond allowed count
        for node, count in list(counter.items()):
            allowed = sync_req.get(node, 2 if node in sync_customers else 1)
            if count > allowed:
                excess = count - allowed
                for rt in self.tours:
                    while node in rt and excess > 0:
                        rt.remove(node)
                        excess -= 1

        # insert missing customers
        for node in non_depot_nodes:
            allowed = sync_req.get(node, 2 if node in sync_customers else 1)
            current = sum(rt.count(node) for rt in self.tours)
            if current < allowed:
                for _ in range(allowed - current):
                    best_v = None
                    best_pos = None
                    best_cost = float('inf')
                    for v in range(len(self.tours)):
                        if self.tours[v].count(node) > 0:
                            continue
                        for pos in range(len(self.tours[v]) + 1):
                            prev = depot_id if pos == 0 else self.tours[v][pos-1]
                            nxt = depot_id if pos == len(self.tours[v]) else self.tours[v][pos]
                            delta = safe_tt(inst.travel_time, prev, node) + safe_tt(inst.travel_time, node, nxt) - safe_tt(inst.travel_time, prev, nxt)
                            if delta < best_cost:
                                best_cost = delta; best_v = v; best_pos = pos
                    if best_v is None:
                        chosen_route = random.randint(0, len(self.tours) - 1)
                        pos = random.randint(0, len(self.tours[chosen_route]))
                        self.tours[chosen_route].insert(pos, node)
                    else:
                        self.tours[best_v].insert(best_pos, node)

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

        # self.eval()
        return self.w1, self.w2
    
    def compute_timeline(self):
        """
        Simulate the solution and return detailed per-vehicle timeline WITHOUT
        changing Solution state. This mirrors eval() simulation but collects
        per-node arrival/start/depart times for inspection.

        Returns a list of dicts, one per vehicle:
        {
          "route": [node ids],
          "visits": [
             {"node": id, "prev": prev, "travel": t_prev_node, "arrival": a, "ready": r,
              "wait": w, "start": s, "service": svc, "depart": d, "lateness": late}
          ],
          "travel": travel_sum_for_vehicle (including return),
          "end_time": end_time (depart_last + return_to_depot)
        }
        """
        inst = self.instance
        nodes = inst.nodes
        TT = inst.travel_time
        depot = min(nodes.keys())

        m = inst.m
        # initialize structure
        timelines = []

        # We'll mirror eval simulation but only to compute and report times
        for k in range(m):
            route = []
            if k < len(self.tours):
                route = list(self.tours[k])
            visits = []
            if not route:
                timelines.append({"route": [], "visits": [], "travel": 0.0, "end_time": 0.0})
                continue

            # simulate vehicle k
            travel_sum = 0.0
            prev = depot
            # arrival at first node = travel from depot to first
            arrival = safe_tt(TT, prev, route[0])
            time_cursor = arrival  # raw arrival (before waiting)
            for idx, node in enumerate(route):
                prev = depot if idx == 0 else route[idx - 1]
                travel_leg = safe_tt(TT, prev, node)
                if idx != 0:
                    time_cursor = depart + travel_leg
                arrival = time_cursor
                ready = nodes[node]["ready"]
                wait = max(0.0, ready - arrival)
                start = arrival + wait
                # lateness is computed relative to due using start
                due = nodes[node]["due"]
                lateness = max(0.0, start - due)
                service = nodes[node]["service"]
                depart = start + service
                visits.append({
                    "node": node,
                    "prev": prev,
                    "travel": float(travel_leg),
                    "arrival": float(arrival),
                    "ready": float(ready),
                    "wait": float(wait),
                    "start": float(start),
                    "service": float(service),
                    "depart": float(depart),
                    "lateness": float(lateness)
                })
                travel_sum += float(travel_leg)

            # add return to depot travel
            last = route[-1]
            ret = safe_tt(TT, last, depot)
            end_time = depart + ret
            travel_sum += float(ret)
            timelines.append({"route": route, "visits": visits, "travel": float(travel_sum), "end_time": float(end_time)})

        return timelines

    def compute_timeline_postsync(self, eps_sync: float = 1e-3):
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
