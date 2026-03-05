import os
import math
from pathlib import Path


# ---------------------------------------------------------
# Data structures
# ---------------------------------------------------------

class Customer:
    def __init__(self, cid, x, y, demand, ready, due, service):
        self.id = int(cid)
        self.x = float(x)
        self.y = float(y)
        self.demand = float(demand)
        self.ready = float(ready)
        self.due = float(due)
        self.service = float(service)


class Instance:
    def __init__(self, name):
        self.name = name
        self.vehicles = 0
        self.n_customers = 0
        self.horizon = 0
        self.synch_ids = []
        self.customers = []
        self.depot = None


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


# ---------------------------------------------------------
# Instance loader
# ---------------------------------------------------------

def load_instance(filepath):

    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip()]

    inst = Instance(Path(filepath).stem)

    inst.vehicles = int(lines[0])
    inst.n_customers = int(lines[1])
    inst.horizon = float(lines[2])
    n_synch = int(lines[3])
    inst.synch_ids = list(map(int, lines[4].split()))

    for line in lines[5:]:

        parts = line.split()

        if len(parts) == 6:
            cid, x, y, ready, due, service = parts
            demand = 0
        else:
            cid, x, y, demand, ready, due, service = parts

        c = Customer(cid, x, y, demand, ready, due, service)

        if c.id == 0:
            inst.depot = c
        else:
            inst.customers.append(c)

    return inst


# ---------------------------------------------------------
# STEP 1 — Basic checks
# ---------------------------------------------------------

def basic_checks(inst):

    if len(inst.customers) != inst.n_customers:
        return False, f"Customer count mismatch: expected {inst.n_customers}, found {len(inst.customers)}"

    if len(inst.synch_ids) == 0:
        return False, "No synchronization customers defined"

    return True, "OK"


# ---------------------------------------------------------
# STEP 2 — Depot reachability
# ---------------------------------------------------------

def depot_reachability(inst):

    depot = inst.depot

    for c in inst.customers:

        trip = dist(depot, c) + c.service + dist(c, depot)

        if trip > inst.horizon:
            return False, f"Customer {c.id} cannot be visited within horizon"

    return True, "OK"


# ---------------------------------------------------------
# STEP 3 — Pairwise compatibility
# ---------------------------------------------------------

def pairwise_compatibility(inst):

    incompatible = []

    for i in inst.customers:
        compatible = False

        for j in inst.customers:
            if i.id == j.id:
                continue

            cond1 = i.ready + i.service + dist(i, j) <= j.due
            cond2 = j.ready + j.service + dist(j, i) <= i.due

            if cond1 or cond2:
                compatible = True
                break

        if not compatible:
            incompatible.append(i.id)

    if incompatible:
        return False, f"Customers with no compatible neighbor: {incompatible}"

    return True, "OK"


# ---------------------------------------------------------
# STEP 4 — Vehicle workload estimation
# ---------------------------------------------------------

def estimate_vehicles(inst):

    depot = inst.depot

    total_service = sum(c.service for c in inst.customers)

    avg_travel = 0
    count = 0

    for c in inst.customers:
        avg_travel += dist(depot, c)
        count += 1

    avg_travel = avg_travel / count

    total_travel_est = 2 * avg_travel * len(inst.customers)

    route_budget = inst.horizon

    vehicles_needed = (total_service + total_travel_est) / route_budget

    vehicles_needed = math.ceil(vehicles_needed)

    if vehicles_needed > inst.vehicles:

        msg = (
            f"Vehicle count too small\n"
            f"estimated vehicles needed = {vehicles_needed}\n"
            f"available = {inst.vehicles}"
        )

        return False, msg

    return True, f"Estimated vehicles needed = {vehicles_needed}"


# ---------------------------------------------------------
# STEP 5 — Synchronization window check
# ---------------------------------------------------------

def synch_check(inst):

    avg_travel = 0
    depot = inst.depot

    for c in inst.customers:
        avg_travel += dist(depot, c)

    avg_travel /= len(inst.customers)

    bad = []

    for cid in inst.synch_ids:

        c = next(x for x in inst.customers if x.id == cid)

        window = c.due - c.ready

        if window < (c.service + avg_travel):
            bad.append(cid)

    if bad:
        return False, f"Synchronization windows too tight for customers {bad}"

    return True, "OK"


# ---------------------------------------------------------
# STEP 6 — Simple constructive heuristic
# ---------------------------------------------------------

def greedy_construct(inst):

    depot = inst.depot
    remaining = set(c.id for c in inst.customers)
    cust_dict = {c.id: c for c in inst.customers}

    vehicles_used = 0

    while remaining:

        vehicles_used += 1

        time = 0
        pos = depot

        while True:

            feasible = []

            for cid in remaining:

                c = cust_dict[cid]

                travel = dist(pos, c)
                arrival = max(time + travel, c.ready)

                if arrival <= c.due:

                    feasible.append((arrival, cid))

            if not feasible:
                break

            feasible.sort()
            _, cid = feasible[0]

            c = cust_dict[cid]

            arrival = max(time + dist(pos, c), c.ready)
            time = arrival + c.service
            pos = c

            remaining.remove(cid)

    if vehicles_used > inst.vehicles:

        msg = (
            f"Greedy construction failed\n"
            f"vehicles needed = {vehicles_used}\n"
            f"available = {inst.vehicles}"
        )

        return False, msg

    return True, f"Greedy used {vehicles_used} vehicles"


# ---------------------------------------------------------
# Validator
# ---------------------------------------------------------

def validate_instance(filepath):

    inst = load_instance(filepath)

    print("\n==============================")
    print(f"Instance {inst.name}")
    print("==============================")

    steps = [
        ("STEP 1 basic checks", basic_checks),
        ("STEP 2 depot reachability", depot_reachability),
        ("STEP 3 pairwise compatibility", pairwise_compatibility),
        ("STEP 4 vehicle estimation", estimate_vehicles),
        ("STEP 5 synchronization windows", synch_check),
        ("STEP 6 constructive heuristic", greedy_construct)
    ]

    for name, func in steps:

        ok, msg = func(inst)

        if ok:
            print(f"{name}: OK ({msg})")
        else:
            print(f"{name}: FAIL")
            print(msg)
            return False

    print("Instance seems FEASIBLE\n")
    return True


# ---------------------------------------------------------
# Folder validator
# ---------------------------------------------------------

def validate_folder(folder):

    files = list(Path(folder).glob("*.txt"))

    ok = 0
    fail = 0

    for f in files:

        if validate_instance(f):
            ok += 1
        else:
            fail += 1

    print("\n==============================")
    print("VALIDATION SUMMARY")
    print("==============================")
    print("Total instances:", len(files))
    print("Feasible:", ok)
    print("Failed:", fail)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    folder = "instances/New_synch/150"

    validate_folder(folder)