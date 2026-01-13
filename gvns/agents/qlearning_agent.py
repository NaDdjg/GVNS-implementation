import os
import pickle
import random
import numpy as np
from collections import defaultdict
from typing import Tuple

from .base_agent import BaseAgent

class QTableAgent(BaseAgent):
    def _new_q_row(self):
        return np.zeros(self.n_actions, dtype=np.float32)

    def _new_count_row(self):
        return np.zeros(self.n_actions, dtype=np.int32)

    def __init__(self,
                 n_actions: int ,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 eps_start: float = 0.2,
                 eps_min: float = 0.01,
                 eps_decay: float = 5e-4,
                 use_ucb: bool = False,
                 ucb_start_step: int = 0,
                 seed: int = 0,
                 ucb_c: float = 1.0,
                 debug: bool = False):
        random.seed(seed)
        np.random.seed(seed)

        self.debug = debug

        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # exploration
        self.fail_streak = np.zeros(self.n_actions, dtype=int)
        self.eps = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # ucb controls
        self.use_ucb = use_ucb
        self.ucb_start_step = ucb_start_step
        self.total_steps = 0
        self.counts = defaultdict(self._new_count_row)   # N(s,a)
        self.state_counts = defaultdict(int)   
        self.ucb_c = ucb_c  
        
        # Q-table
        self.Q = defaultdict(self._new_q_row)

    # discretization helper
    # we use semantic buckets not raw values, because we don't want the agent to memorize numbers, but recongnize situations
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
          8 -> stagnation
          9 -> elapsed_time
          10-13 -> neigh success rates

        returns tuple of ints to index the Q-table.
        """
        # unpack
        penS_flag = 1 if obs[7] > 1e-9 else 0
        penT_flag = 1 if obs[5] > 1e-9 else 0
        penH_flag = 1 if obs[6] > 1e-9 else 0

        # cont bucket 
        # to learn which neighborhoods work better when search is stuck
        stagn  = int(obs[8])
        if stagn  == 0:
            stag_bucket  = 0     # just improved
        elif stagn <= 2:
            stag_bucket = 1      # short stagnation
        elif stagn <= 6:
            stag_bucket = 2      # medium
        else:
            stag_bucket = 3      # long stagnation

        # neighborhood effectiveness proxy 
        # success rate when chosen as first improving neighborhood (how successful neighborhoods have been recently)
        best_succ = max(obs[-self.n_actions:])  # best success rate among neighbors as proxy 
        if best_succ > 0.6:
            eff_bucket = 0   # strong
        elif best_succ > 0.2:
            eff_bucket = 1   # moderate
        elif best_succ > 0.0:
            eff_bucket = 2   # weak
        else:
            eff_bucket = 3   # dead

        # instance size
        n = int(obs[0])
        if n < 100:
            size_bucket = 0
        elif n < 200:
            size_bucket = 1
        else:
            size_bucket = 2

        # return state key tuple 
        # information: feasibility, is the search stuck, is the region promising, instance size
        return (
            penS_flag,
            penT_flag,
            penH_flag,
            stag_bucket,
            eff_bucket,
            size_bucket
        )

    # BaseAgent API
    # ============================================================
    # Action selection
    # ============================================================
    def select_action(self, obs: np.ndarray) -> int:
        """
        given an observation (raw vector), discretize and pick action via epsilon-greedy.
        """
        s = self.discretize_obs(obs)
        self.total_steps += 1

        #  UCB (after warm-up) 
        if self.use_ucb and self.total_steps >= self.ucb_start_step:
            # ensure each action is tried at least once
            untried = np.where(self.counts[s] == 0)[0]

            if len(untried) > 0:
                return int(np.random.choice(untried))

            qvals = self.Q[s]
            n_s = self.state_counts[s]
            n_sa = self.counts[s] + 1e-9

            bonus = self.ucb_c * np.sqrt(np.log(n_s) / n_sa)
            return int(np.argmax(qvals + bonus))
        
        # --- epsilon-greedy ---
        if random.random() < self.eps:
            return random.randrange(self.n_actions)

        return int(np.argmax(self.Q[s]))

    # ============================================================
    # Q update
    # ============================================================
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

        # past experience
        q_old  = self.Q[s][action] 
        # does this choice lead to a promising search region
        q_next  = 0.0 if done else float(np.max(self.Q[s_next]))  

        target = reward + self.gamma * q_next 
        # the updated belief about this neighborhood in this state
        q_new = q_old + self.alpha * (target - q_old)

        self.Q[s][action] = q_new # expected improvement if the agent try this neighborhood now
        
        # increment visit counts for UCB bookkeeping
        self.counts[s][action] += 1
        self.state_counts[s] += 1

        # decay epsilon a bit (only when not using ucb yet)
        if not self.use_ucb and self.eps > self.eps_min:
            self.eps = max(self.eps_min, self.eps - self.eps_decay)

        if self.debug:
            print(
                f"[Q-UPDATE] state={s} | "
                f"action={action} | "
                f"Q_old={q_old:.4f} | "       
                f"reward={reward:.4f} | "
                f"Q_next_max={q_next:.4f} | "
                f"Q_new={q_new:.4f}"
            )

    # ============================================================
    # persistence
    # ============================================================
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "Q": dict(self.Q),
            "counts": dict(self.counts),
            "state_counts": dict(self.state_counts),
            "eps": self.eps,
            "total_steps": self.total_steps,
            "params": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "n_actions": self.n_actions,
                "use_ucb": self.use_ucb,
                "ucb_c": self.ucb_c
            }
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)


    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
    
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32),
            data["Q"]
        )
        self.counts = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.int32),
            data.get("counts", {})
        )
        self.state_counts = defaultdict(int, data.get("state_counts", {}))
    
        self.eps = data.get("eps", self.eps)
        self.total_steps = data.get("total_steps", 0)
