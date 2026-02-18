import math

class Instance:
    """
    data container for a VRPTW-S 
    """
    def __init__(self):
        self.instance_type = None        # "liu" or "bredstrom"
        self.m = None                    # number of vehicles
        self.n = None                    # number of customers
        self.horizon = None              # global planning horizon
        self.nodes = {}                  # id -> {x, y, ready, due, service}
        self.sync_customers = set()      # customers requiring multiple vehicles
        self.sync_requirements = {}      # customer_id -> required_vehicles (usually 2)
        self.depot_id = 0                # assuming depot is always node 0 by default
        self.travel_time = None          # matrix of distances
        self.worker_cost = None          # worker-service cost matrix 
        self.speed = 1.0
        
    @staticmethod  
    def _try_parse_ints(line: str):
        try:
            return [int(x) for x in line.strip().split()]
        except:
            return None
        
    def normalize_synchronization(self):
        """
        Canonicalize synchronization so that the instance behaves
        exactly like a Liu instance everywhere else.

        - Liu instances: already normalized → do nothing
        - Bredström instances: collapse synchronized duplicate nodes
        """

        # case 1: Liu-style (already canonical)
        if getattr(self, "sync_customers", None):
            return

        # case 2: Bredström-style (pairwise duplicates)
        if not hasattr(self, "sync_pairs") or not self.sync_pairs:
            return

        self.sync_customers = set()
        self.sync_requirements = {}

        # build alias mapping
        aliases = set()

        for (i, j) in self.sync_pairs:
            if i == self.depot_id or j == self.depot_id:
                raise ValueError("Depot cannot be part of a synchronization pair")

            rep = min(i, j)
            alias = max(i, j)

            self.sync_customers.add(rep)
            self.sync_requirements[rep] = 2
            aliases.add(alias)

        aliases = sorted(aliases)

        # determine nodes to keep
        old_ids = sorted(i for i in self.nodes.keys() if i not in aliases)

        # old → new contiguous ID mapping
        id_map = {old: new for new, old in enumerate(old_ids)}

        # rebuild travel time matrix 
        old_tt = self.travel_time
        new_tt = []

        for i_old in old_ids:
            row = [old_tt[i_old][j_old] for j_old in old_ids]
            new_tt.append(row)

        self.travel_time = new_tt

        # rebuild nodes
        self.nodes = {id_map[i]: self.nodes[i] for i in old_ids}
        self.depot_id = id_map[self.depot_id]

        # rebuild worker cost matrix if present
        if getattr(self, "worker_cost", None) is not None:
            self.worker_cost = {
                id_map[i]: self.worker_cost[i]
                for i in self.worker_cost
                if i in id_map
            }

        # rebuild synch structures
        self.sync_customers = {id_map[i] for i in self.sync_customers}
        self.sync_requirements = {
            id_map[i]: req for i, req in self.sync_requirements.items()
        }

        # cleanup
        del self.sync_pairs

    def load_liu_format(self, path: str):
        """
        parser for Liu benchmark format instances
        the header looks like:
            30
            100
            3390
            5
            18 30 61 76 88
        followed by node rows:
            id  x  y  ready  due  service
        """
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        # header
        header = []
        idx = 0
        while idx < len(lines) and len(header) < 4:
            nums = self._try_parse_ints(lines[idx])
            if nums:
                header.extend(nums)
            idx += 1

        if len(header) < 4:
            raise ValueError("Header incomplete (need m, n, horizon, s).")

        self.m = header[0]
        self.n = header[1]
        self.horizon = float(header[2])
        s = header[3]  # number of synchronized customers

        # synchronized customer IDs
        sync_ids = []
        while len(sync_ids) < s and idx < len(lines):
            nums = self._try_parse_ints(lines[idx])
            if nums:
                sync_ids.extend(nums)
            idx += 1

        self.sync_customers = set(sync_ids[:s])
        self.sync_requirements = {cid: 2 for cid in self.sync_customers}

        # remaining lines: nodes
        node_lines = lines[idx:]
        for ln in node_lines:
            toks = ln.split()
            if len(toks) != 6:
                continue
            i, x, y, ready, due, serv = toks
            i = int(i)
            self.nodes[i] = {
                "id": i,
                "x": float(x),
                "y": float(y),
                "ready": float(ready),
                "due": float(due),
                "service": float(serv),
            }

        # depot 
        min_id = min(self.nodes.keys()) 
        for nid, info in self.nodes.items():
            if info.get("ready", None) == 0 and info.get("service", None) == 0:
                min_id = nid
                break
        self.depot_id = min_id

        # build travel time matrix (euclidean)
        max_id = max(self.nodes.keys())
        size = max_id + 1
        self.travel_time = [[0.0]*size for _ in range(size)]
        for i in self.nodes:
            xi, yi = self.nodes[i]['x'], self.nodes[i]['y']
            for j in self.nodes:
                if i == j: continue
                xj, yj = self.nodes[j]['x'], self.nodes[j]['y']
                dist = math.hypot(xi - xj, yi - yj)
                try:
                    tt = dist / float(self.speed) if float(self.speed) > 0 else dist
                except Exception:
                    tt = dist
                self.travel_time[i][j] = tt

        return self

    def load_bredstrom_format(self, path: str):
        """
        Parser for Bredström VRPTW-S instances.

        Sections are separated by blank lines:

        Section 1: Metadata
        Section 2: Time windows (ready, due)
        Section 3: Service durations
        Section 4: Travel time matrix
        Section 5: Worker-service cost matrix 
        """
        # Read file and split into sections
        with open(path, "r") as f:
            raw_lines = [l.rstrip() for l in f]

        sections = []
        current = []

        for line in raw_lines:
            if line.strip() == "":
                if current:
                    sections.append(current)
                    current = []
            else:
                current.append(line.strip())

        if current:
            sections.append(current)

        if len(sections) < 4:
            raise ValueError("Invalid Bredström file: missing sections")

        # --------------------------------------------------
        # SECTION 1 — METADATA
        # --------------------------------------------------
        meta = sections[0]

        # line 0: instance name (ignored)
        # line 1: number of vehicles
        # line 2: number of customers
        # line 3: horizon
        # line 4: number of synchronized pairs
        self.m = int(meta[1])
        self.n = int(meta[2])
        self.horizon = float(meta[3])

        n_sync = int(meta[4])

        # read synchronized pairs
        sync_pairs = []
        for i in range(n_sync):
            a, b = map(int, meta[5 + i].split())
            sync_pairs.append((a, b))

        self.sync_pairs = sync_pairs

        # --------------------------------------------------
        # SECTION 2 — TIME WINDOWS
        # --------------------------------------------------
        # format: node_id  ready  due
        self.nodes = {}

        for line in sections[1]:
            nid, ready, due = line.split()
            nid = int(nid)

            self.nodes[nid] = {
                "id": nid,
                "x": 0.0,            # not used in Bredström
                "y": 0.0,            # not used in Bredström
                "ready": float(ready),
                "due": float(due),
                "service": 0.0       # filled next
            }

        # depot is explicitly node 0
        self.depot_id = 0

        # --------------------------------------------------
        # SECTION 3 — SERVICE DURATIONS
        # --------------------------------------------------
        # format: node_id  service_time
        for line in sections[2]:
            nid, service = line.split()
            nid = int(nid)

            if nid not in self.nodes:
                raise ValueError(f"Service time for unknown node {nid}")

            self.nodes[nid]["service"] = float(service)

        # --------------------------------------------------
        # SECTION 4 — TRAVEL TIME MATRIX
        # --------------------------------------------------
        matrix_lines = sections[3]

        # number of nodes inferred from rows
        size = len(matrix_lines)
        self.travel_time = [[0.0] * size for _ in range(size)]

        for row_idx, line in enumerate(matrix_lines):
            toks = line.split()

            # first token is the row id
            row_id = int(toks[0])
            values = list(map(float, toks[1:]))

            if len(values) != size:
                raise ValueError(
                    f"Travel time row {row_id} has {len(values)} values, expected {size}"
                )

            for j, v in enumerate(values):
                self.travel_time[row_id][j] = v
        
        last_id = len(self.travel_time) - 1
        if self.travel_time[0] == self.travel_time[last_id]:
            # remove last row
            self.travel_time.pop()

            # remove last column
            for row in self.travel_time:
                row.pop()

            # remove node entry if present
            if last_id in self.nodes:
                self.nodes.pop(last_id)

        # --------------------------------------------------
        # SECTION 5 — COST MATRIX (worker preference)
        # --------------------------------------------------
        if len(sections) >= 5:
            cost_lines = sections[4]

            # index by: cost[service_id][worker_id]
            worker_utility = {}

            for line in cost_lines:
                toks = line.split()
                service_id = int(toks[0])
                costs = list(map(float, toks[1:]))

                worker_utility[service_id] = costs

            self.worker_cost = worker_utility

        self.normalize_synchronization()

        # update counters
        self.n = sum(1 for nid in self.nodes if nid != self.depot_id)

        # sanity check
        expected = len(self.travel_time) - 1
        assert self.n == expected, (
            f"Inconsistent instance: n={self.n}, matrix={len(self.travel_time)}"
        )

        return self
    
    @staticmethod
    def parse_instance(path: str):
        """
        Global parser that dispatches to the correct loader
        based on the instance path.
        """
        inst = Instance()

        norm_path = path.replace("\\", "/")

        if "instances/Liu_ALNS/" in norm_path:
            inst.instance_type = "liu"
            inst.load_liu_format(path)

        elif "instances/Bredstrom/" in norm_path:
            inst.instance_type = "bredstrom"
            inst.load_bredstrom_format(path)

        else:
            raise ValueError(f"Unknown instance format for path: {path}")

        return inst

    def __repr__(self):
        return (f"<Instance n={self.n} m={self.m} horizon={self.horizon} "
                f"sync_customers={list(self.sync_customers)}>")

    def debug_dump(self):
        print("\n" + "=" * 80)
        print("INSTANCE DEBUG DUMP")
        print("=" * 80)

        # --------------------------------------------------
        # basic metadata
        # --------------------------------------------------
        print("\n[METADATA]")
        print(f"  Vehicles (m): {self.m}")
        print(f"  Customers (n): {self.n}")
        print(f"  Horizon: {self.horizon}")
        print(f"  Depot ID: {self.depot_id}")

        # --------------------------------------------------
        # synchronization
        # --------------------------------------------------
        print("\n[SYNCHRONIZATION]")
        print(f"  Sync customers: {sorted(self.sync_customers)}")
        print(f"  Sync requirements:")
        for k, v in self.sync_requirements.items():
            print(f"    Customer {k} -> {v} vehicles")

        # --------------------------------------------------
        # nodes
        # --------------------------------------------------
        print("\n[NODES]")
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            print(
                f"  Node {nid}: "
                f"ready={n['ready']}, due={n['due']}, "
                f"service={n['service']}, "
                f"x={n['x']}, y={n['y']}"
            )

        # --------------------------------------------------
        # TRAVEL TIME MATRIX
        # --------------------------------------------------
        print("\n[TRAVEL TIME MATRIX]")
        print(f"  Size: {len(self.travel_time)} x {len(self.travel_time[0])}")
        for i, row in enumerate(self.travel_time):
            row_str = " ".join(f"{v:7.1f}" for v in row)
            print(f"  {i:2d}: {row_str}")

        # --------------------------------------------------
        # WORKER COST MATRIX
        # --------------------------------------------------
        print("\n[WORKER COST MATRIX]")
        if hasattr(self, "worker_cost") and self.worker_cost:
            for service_id in sorted(self.worker_cost):
                costs = self.worker_cost[service_id]
                cost_str = " ".join(f"{c:7.1f}" for c in costs)
                print(f"  Service {service_id}: {cost_str}")
        else:
            print("  <none>")

        print("\n" + "=" * 80)