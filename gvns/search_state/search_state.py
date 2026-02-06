import time
from collections import deque
from typing import Dict, Any

class SearchState:
    """
    tracks instance-level, search-level, and neighborhood-context features
    for FI / BI decision making.
    """

    def __init__(
        self,
        instance,
        time_limit: float,
        window_size: int = 20  # parameter that answers "what has ls been doing recently?"
    ):
        self.instance = instance
        self.time_limit = float(time_limit)
        self.W = int(window_size)

        #  static (instance-level) 
        self._extract_instance_features()

        #  dynamic (search-level) 
        self.start_time = None
        self.iter_gvns = 0
        self.ls_calls_since_improvement = 0

        # rolling LS history
        self.improved_hist = deque(maxlen=self.W)
        self.delta_hist = deque(maxlen=self.W)
        self.elapsed_hist = deque(maxlen=self.W)

        # current solution snapshot (for penalties)
        self.current_solution = None

    # =========================================================
    # Initialization helpers
    # =========================================================
    def _extract_instance_features(self):
        """Compute all instance-level features once."""

        inst = self.instance
        nodes = inst.nodes

        self.inst_features = {}

        # core
        self.inst_features["n_customers"] = inst.n
        self.inst_features["n_vehicles"] = inst.m
        self.inst_features["horizon"] = inst.horizon
        self.inst_features["sync_ratio"] = (
            len(inst.sync_customers) / inst.n if inst.n > 0 else 0.0
        )

        # time window structure
        tw_widths = []
        early = 0
        late = 0

        for i, node in nodes.items():
            ready = node["ready"]
            due = node["due"]
            width = max(0.0, due - ready)
            tw_widths.append(width)

            if ready < 0.2 * inst.horizon:
                # early time window , first 20% of horizon
                early += 1
            if due > 0.8 * inst.horizon:
                # late time window , last 20% of horizon
                late += 1

        if tw_widths:
            avg_tw = sum(tw_widths) / len(tw_widths)
            min_tw = min(tw_widths)
        else:
            avg_tw = 0.0
            min_tw = 0.0

        self.inst_features.update({
            "avg_tw_width": avg_tw,
            "min_tw_width": min_tw,
            "tw_tightness": avg_tw / inst.horizon if inst.horizon > 0 else 0.0,
            "early_tw_ratio": early / inst.n if inst.n > 0 else 0.0,
            "late_tw_ratio": late / inst.n if inst.n > 0 else 0.0,
        })

    # =========================================================
    # Reset 
    # =========================================================
    def reset(self):
        """reset all dynamic search-level statistics."""
        self.start_time = time.time()
        self.iter_gvns = 0
        self.ls_calls_since_improvement = 0

        self.improved_hist.clear()
        self.delta_hist.clear()
        self.elapsed_hist.clear()

        self.current_solution = None

    # =========================================================
    # Update functions (called by local search)
    # =========================================================
    def before_ls(self, k: int, iter_gvns: int):
        self.iter_gvns = iter_gvns
        self.current_k = k

    def after_ls(
        self,
        improved: bool,
        F_before: float,
        F_after: float,
        elapsed: float,
        solution
    ):
        delta = max(0.0, F_before - F_after)

        # update rolling history
        self.improved_hist.append(1 if improved else 0)
        self.delta_hist.append(delta)
        self.elapsed_hist.append(elapsed)

        # stagnation counter
        if improved:
            self.ls_calls_since_improvement = 0
        else:
            self.ls_calls_since_improvement += 1

        # update current solution snapshot
        self.current_solution = solution

    # =========================================================
    # Feature extraction
    # =========================================================
    def get_features(self, k: int, k_max: int) -> Dict[str, Any]:
        """
        return full feature dictionary:
        [instance] + [search] + [neighborhood context]
        """
        feats = {}
        # instance 
        feats.update(self.inst_features)

        # search 
        now = time.time()
        elapsed = now - self.start_time if self.start_time else 0.0

        time_elapsed_ratio = (
            elapsed / self.time_limit if self.time_limit > 0 else 0.0
        )

        ls_phase_norm = (
            self.ls_calls_since_improvement /
            (self.ls_calls_since_improvement + self.W)
        )

        feats.update({
            "iter_gvns": self.iter_gvns,
            "stagnation_ls": self.ls_calls_since_improvement,
            "ls_phase_norm": ls_phase_norm,
            "time_elapsed_ratio": min(1.0, time_elapsed_ratio),
            "time_left_ratio": max(0.0, 1.0 - time_elapsed_ratio),
        })

        # rolling-window stats
        W_eff = len(self.improved_hist)
        if W_eff > 0:
            improv_rate = sum(self.improved_hist) / W_eff
            avg_delta = sum(self.delta_hist) / W_eff
            max_delta = max(self.delta_hist)
        else:
            improv_rate = 0.0
            avg_delta = 0.0
            max_delta = 0.0

        feats.update({
            "improv_rate_W": improv_rate,
            "avg_delta_W": avg_delta,
            "max_delta_W": max_delta,
            "ls_fail_ratio_W": 1.0 - improv_rate,
        })

        #  penalty structure 
        sol = self.current_solution
        if sol is None or sol.Fval == 0:
            feats.update({
                "PV": 0,
                "pen_ratio": 0.0,
                "penT_ratio": 0.0,
                "penS_ratio": 0.0,
            })
        else:
            total_pen = sol.penT + sol.penH + sol.penS
            feats.update({
                "PV": sol.PV,
                "pen_ratio": total_pen / sol.Fval,
                "penT_ratio": sol.penT / sol.Fval,
                "penS_ratio": sol.penS / sol.Fval,
            })

        #  neighborhood context 
        feats.update({
            "k": k,
            "k_norm": k / k_max if k_max > 0 else 0.0,
        })

        return feats

    def get_model_features(self, k: int, k_max: int) -> Dict[str, Any]:
        raw = self.get_features(k, k_max)
        mapped = {}

        FEATURE_MAP = {
            # direct
            "gvns_iter": "iter_gvns",
            "stagnation_ls": "stagnation_ls",
            "time_elapsed_ratio": "time_elapsed_ratio",
            "time_left_ratio": "time_left_ratio",
            "improv_rate_W": "improv_rate_W",
            "avg_delta_W": "avg_delta_W",
            "max_delta_W": "max_delta_W",
            "ls_fail_ratio_W": "ls_fail_ratio_W",
            "PV": "PV",
            "pen_ratio": "pen_ratio",
            "penT_ratio": "penT_ratio",
            "penS_ratio": "penS_ratio",
            "k": "k",
            "k_norm": "k_norm",

            # instance
            "n_customers": "n_customers",
            "n_vehicles": "n_vehicles",
            "horizon": "horizon",
            "sync_ratio": "sync_ratio",
            "avg_tw_width": "avg_tw_width",
            "min_tw_width": "min_tw_width",
            "tw_tightness": "tw_tightness",
            "early_tw_ratio": "early_tw_ratio",
            "late_tw_ratio": "late_tw_ratio",
        }

        for model_name, raw_name in FEATURE_MAP.items():
            mapped[model_name] = raw.get(raw_name, 0.0)

        # features that must exist but are not decision-time known
        mapped["elapsed"] = raw.get("time_elapsed_ratio", 0.0) * self.time_limit
        mapped["neigh_id"] = k

        return mapped
