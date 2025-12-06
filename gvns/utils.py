import numpy as np, time

NEIGHBORHOOD_NAMES = ['N1','N2','N3','N4','N5','N6','N7','N8']
NEIGHBORHOODS = NEIGHBORHOOD_NAMES

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