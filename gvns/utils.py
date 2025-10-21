from .evaluate import evaluate_solution
from .viz import visualize_solution
import numpy as np, time

NEIGHBORHOODS = ['N1','N2','N3','N4']

# clssic utils 

import random
from typing import List, Tuple, Optional

def add_sync_pairs(inst, num_sync_visits: int, seed: Optional[int] = None, avoid_duplicate_nodes: bool = True) -> List[Tuple[int,int]]:
    """
    Add synchronization pairs to the instance `inst` using `num_sync_visits` visits.
    Each pair is formed from 2 visits, so the number of pairs = num_sync_visits // 2.
    - inst.visits is expected to be a mapping (keys = visit ids).
    - The function sets inst.sync_pairs to a list of unordered tuples (i, j).
    - By default, we avoid a node appearing in >1 pair (avoid_duplicate_nodes=True).
    - Returns the list of pairs added.
    """
    if seed is not None:
        random.seed(seed)

    visit_ids = sorted(list(getattr(inst, "visits", {}).keys()))

    n = len(visit_ids)
    if n < 2 or num_sync_visits < 2:
        inst.sync_pairs = []
        return []

    # convert visits to pairs
    num_pairs = num_sync_visits // 2
    max_pairs_possible = n // 2 if avoid_duplicate_nodes else (n * (n - 1)) // 2
    if num_pairs > max_pairs_possible:
        raise ValueError(
            f"Requested {num_pairs} pairs ({num_sync_visits} visits) "
            f"but only {max_pairs_possible} pairs possible (n={n})."
        )

    pairs = []
    used = set()

    attempts = 0
    max_attempts = 10_000
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        if avoid_duplicate_nodes:
            remaining = [v for v in visit_ids if v not in used]
            if len(remaining) < 2:
                break
            a, b = random.sample(remaining, 2)
        else:
            a, b = random.sample(visit_ids, 2)

        if a == b:
            continue
        if a > b:
            a, b = b, a
        if (a, b) in pairs:
            continue

        pairs.append((a, b))
        if avoid_duplicate_nodes:
            used.add(a)
            used.add(b)

    if len(pairs) < num_pairs:
        raise RuntimeError(
            f"Could not generate requested number of sync pairs ({num_pairs}); "
            f"produced {len(pairs)} after {attempts} attempts."
        )

    inst.sync_pairs = pairs
    return pairs


def _maybe_get_inst_and_sol_from_args(args, kwargs):
    if len(args) >= 2:
        return args[0], args[1]
    elif 'inst' in kwargs and 'sol' in kwargs:
        return kwargs['inst'], kwargs['sol']
    return None, None

def wrap_with_visualization(func_name, before=True, after=True, debug_visual=True, max_nodes=60):
    if func_name not in globals():
        print(f"[wrap] {func_name} not found — skip")
        return False
    orig = globals()[func_name]
    if not callable(orig):
        print(f"[wrap] {func_name} not callable — skip")
        return False

    def wrapped(*args, **kwargs):
        inst, sol = _maybe_get_inst_and_sol_from_args(args, kwargs)
        if debug_visual and inst and sol:
            try:
                ev = evaluate_solution(inst, sol)
            except Exception:
                ev = {}
            nodes = sum(len(r) for r in sol.routes)
            if nodes <= max_nodes and before:
                visualize_solution(inst, sol, start_times=ev.get('start_times'), title=f"{func_name}_before", show=True)
        res = orig(*args, **kwargs)
        new_sol = None
        if isinstance(res, tuple):
            for item in res:
                if hasattr(item, 'routes'):
                    new_sol = item
                    break
        elif hasattr(res, 'routes'):
            new_sol = res
        else:
            new_sol = args[1] if len(args)>=2 else kwargs.get('sol', None)

        if debug_visual and inst and new_sol and after:
            try:
                ev2 = evaluate_solution(inst, new_sol)
            except Exception:
                ev2 = {}
            nodes2 = sum(len(r) for r in new_sol.routes)
            if nodes2 <= max_nodes:
                visualize_solution(inst, new_sol, start_times=ev2.get('start_times'), title=f"{func_name}_after", show=True)
        return res

    globals()[func_name] = wrapped
    print(f"[wrap] {func_name} wrapped for visualization (before={before}, after={after})")
    return True

def infer_vehicle_estimate(default=60) -> int:
    """
    i'll keep it static following the paper values for now
    """
    return default

def pick_vehicle_avoiding_partners(VF, m, partners):
    # Return a vehicle index in 0..m-1 that is not assigned to any partner already.
    available = [v for v in range(m) if all(VF.get(p, -1) != v for p in partners)]
    if available:
        return random.choice(available)
    # if none available, return random (or extend fleet)
    return random.randrange(m)

# rl utils
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = 1.0 if x.ndim == 1 else x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        # Welford-style online update of mean and variance
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 10.0):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -clip, clip)

def extract_state(inst, current_sol, eval_current, iter_no, start_time=None, neigh_success=None, neigh_attempts=None):
    n = len(inst.visits)
    tw_widths = [v.tw_late - v.tw_early for v in inst.visits.values()]
    avg_tw = float(np.mean(tw_widths)) if len(tw_widths) > 0 else 0.0
    sync_ratio = len(getattr(inst, "sync_pairs", [])) / max(1, n)

    feats = [float(n), avg_tw, sync_ratio]

    evalf = [float(eval_current['F']), float(eval_current['f']),
             float(eval_current.get('PenT', 0.0)),
             float(eval_current.get('PenH', 0.0)),
             float(eval_current.get('PenS', 0.0))]

    progress = [float(iter_no), float(time.time() - (start_time or 0.0))]

    succ_rates = []
    if neigh_success is not None and neigh_attempts is not None:
        for s, a in zip(neigh_success, neigh_attempts):
            succ_rates.append(s / a if a > 0 else 0.0)
    else:
        succ_rates = [0.0] * len(NEIGHBORHOODS)

    succ_rates = [float(np.clip(x, 0.0, 1.0)) for x in succ_rates]
    
    return np.array(feats + evalf + progress + succ_rates, dtype=np.float32)

def obs_dim(n_neigh: int = None) -> int:
    """return observation vector dimension: (3 instance feats) + (5 eval) + (2 progress) + n_neigh"""
    if n_neigh is None:
        n_neigh = len(NEIGHBORHOODS)
    return 3 + 5 + 2 + n_neigh