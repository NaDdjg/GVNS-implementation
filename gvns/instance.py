# gvns/instance.py
from typing import List, Dict
import math

class Instance:
    """
    Data container for a VRPTW-S (multi-vehicle synchronization) instance.
    """

    def __init__(self):
        self.m = None          # number of vehicles
        self.n = None          # number of customers
        self.horizon = None    # global planning horizon
        self.nodes = {}        # id -> {x, y, ready, due, service}
        self.sync_customers = set()      # customers requiring multiple vehicles
        self.sync_requirements = {}      # customer_id -> required_vehicles (usually 2)
        self.travel_time = None          # matrix of distances

    @staticmethod
    def _try_parse_ints(line: str):
        try:
            return [int(x) for x in line.strip().split()]
        except:
            return None

    def load_liu_format(self, path: str):
        """
        Parser for Liu benchmark format used by Masmoudi et al.
        Example header:
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

        # next line(s): synchronized customer IDs
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

        # build travel time matrix (Euclidean)
        max_id = max(self.nodes.keys())
        size = max_id + 1
        self.travel_time = [[0.0]*size for _ in range(size)]
        for i in self.nodes:
            xi, yi = self.nodes[i]['x'], self.nodes[i]['y']
            for j in self.nodes:
                if i == j: continue
                xj, yj = self.nodes[j]['x'], self.nodes[j]['y']
                self.travel_time[i][j] = math.hypot(xi - xj, yi - yj)

        return self

    def __repr__(self):
        return (f"<Instance n={self.n} m={self.m} horizon={self.horizon} "
                f"sync_customers={list(self.sync_customers)}>")
