import os
import pickle
import random
import numpy as np
from collections import defaultdict
from typing import Tuple, Any, Dict

from .base_agent import BaseAgent

# =============================================================================
# QTableAgent: Tabular Q-learning for neighborhood selection
# =============================================================================
#
# Design notes:
# - Uses a discretizer to convert BVNDEnv observation vector into a compact state_key tuple.
# - Q is stored as a dict: Q[state_key] -> numpy array (n_actions,)
# - Designed for online updates: call select_action(obs) -> env.step(action) -> update(...)
#
# Mapping to architecture:
# - Discretizer -> "Discretizer" box
# - Q-table -> "Q-Table" box
# - Epsilon-Greedy -> "Action selection" box
# =============================================================================

class QTableAgent(BaseAgent):
    def __init__(self,
                 n_actions: int = 4,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 eps_start: float = 0.2,
                 eps_min: float = 0.01,
                 eps_decay: float = 1e-5,
                 discretizer_config: Dict = None,
                 seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))
        # discretizer_config can hold thresholds/bins; we'll use defaults if None
        self.disc_cfg = discretizer_config or {}

    # discretization helper
    def discretize_obs(self, obs: np.ndarray) -> Tuple:
        """
        convert observation vector produced by BVNDEnv._make_state() into a discrete key.
        expected obs shape: [ n, avg_tw, sync_ratio, F, f, PenT, PenH, PenS, iter, elapsed, neigh_succ_0..3 ]
        indx mapping:
          0 -> n_customers
          1 -> avg_tw
          2 -> sync_ratio
          3 -> F
          4 -> f
          5 -> PenT
          6 -> PenH
          7 -> PenS
          8 -> iter / step_count
          9 -> elapsed_time
          10-13 -> neigh success rates

        returns tuple of ints to index the Q-table.
        """
        # unpack
        n = int(obs[0])
        penT_flag = 1 if obs[5] > 1e-9 else 0
        penH_flag = 1 if obs[6] > 1e-9 else 0

        # cont bucket : 0 => fresh, 1=> small, 2=> large
        iter_val = int(obs[8])
        if iter_val == 0:
            cont_bucket = 0
        elif iter_val <= 3:
            cont_bucket = 1
        else:
            cont_bucket = 2

        # last improvement magnitude - here we use neigh success: if any success > 0.5 -> good
        last_impr = max(obs[10:14])  # best success rate among neighbors as proxy
        if last_impr > 0.5:
            delta_bucket = 0  # good
        elif last_impr > 0.0:
            delta_bucket = 1  # small
        else:
            delta_bucket = 2  # none

        # size bucket
        if n < 25:
            size_bucket = 0
        elif n < 50:
            size_bucket = 1
        else:
            size_bucket = 2

        # state key tuple
        key = (penT_flag, penH_flag, cont_bucket, delta_bucket, size_bucket)
        return key

    # BaseAgent API
    def select_action(self, obs: np.ndarray) -> int:
        """
        given an observation (raw vector), discretize and pick action via epsilon-greedy.
        """
        key = self.discretize_obs(obs)
        if random.random() < self.eps:
            a = random.randrange(self.n_actions)
        else:
            a = int(np.argmax(self.Q[key]))
        return a

    def update(self,
               obs: np.ndarray,
               action: int,
               reward: float,
               next_obs: np.ndarray,
               done: bool):
        """
        online Q update using the Bellman equation.
        """
        s = self.discretize_obs(obs)
        s_next = self.discretize_obs(next_obs)
        q_sa = self.Q[s][action]
        q_next_max = 0.0 if done else float(np.max(self.Q[s_next]))
        target = reward + self.gamma * q_next_max
        self.Q[s][action] = q_sa + self.alpha * (target - q_sa)

        # decay epsilon a bit
        if self.eps > self.eps_min:
            self.eps = max(self.eps_min, self.eps - self.eps_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            # convert defaultdict to normal dict for pickle
            pickle.dump(dict(self.Q), f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32), data)
