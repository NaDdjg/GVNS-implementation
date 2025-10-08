import time
import numpy as np
import random
import gym
from gym import spaces
from typing import Optional, Tuple, Dict, Any

from .operators import clone_solution
from .evaluate import evaluate_solution
from .local_search import b_vnd
from .utils import NEIGHBORHOODS, extract_state, obs_dim

class BVNDEnv(gym.Env):
    """
    gym environment wrapper around one-decision-in-B-VND.
    action: choose neighborhood index (0..3)
    observation: concatenation of scalar instance+solution+search stats + neigh success rates
    """

    metadata = {"render.modes": []}

    def __init__(self, inst, initial_sol,
                 w1: float = 0.5, w2: float = 0.1, M: float = 1e6,
                 Tmax: float = 2000.0, b_vnd_max_iters: int = 2000,
                 max_steps: int = 100, time_penalty_coef: float = 1e-3,
                 seed: Optional[int] = None):
        super().__init__()
        self.inst = inst
        self.initial_sol = initial_sol
        self.w1 = w1
        self.w2 = w2
        self.M = M
        self.Tmax = float(Tmax)
        self.b_vnd_max_iters = int(b_vnd_max_iters)
        self.max_steps = int(max_steps)
        self.time_penalty_coef = float(time_penalty_coef)

        # track attempts + successes per neighborhood
        self.neigh_attempts = [0.0] * len(NEIGHBORHOODS)
        self.neigh_success = [0.0] * len(NEIGHBORHOODS)

        # build observation space dynamically (here i use use 14 dims: 3 + 5 + 2 + 4)
        self._obs_dim = obs_dim(len(NEIGHBORHOODS))
        low = -np.inf * np.ones(self._obs_dim, dtype=np.float32)
        high = np.inf * np.ones(self._obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.action_space = spaces.Discrete(len(NEIGHBORHOODS))

        # RNG
        self.seed(seed)

        # internal state
        self.current = None
        self.eval = None
        self.best_F = None
        self.start_time = None
        self.step_count = 0
        self.last_improved_step = 0

        # initialize
        self.reset()

    def seed(self, seed: Optional[int] = None):
        self._seed = seed if seed is not None else random.randrange(2**30)
        random.seed(self._seed)
        np.random.seed(self._seed)
        return [self._seed]

    # state extraction and feature vector part 
    def _make_state(self) -> np.ndarray:
        return extract_state(
            self.inst, self.current, self.eval,
            iter_no=self.step_count, start_time=self.start_time,
            neigh_success=self.neigh_success, neigh_attempts=self.neigh_attempts
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        # reset solution and metrics
        self.current = clone_solution(self.initial_sol)
        self.eval = evaluate_solution(self.inst, self.current, self.w1, self.w2, self.M)
        self.best_F = float(self.eval['F'])
        self.start_time = time.time()
        self.step_count = 0
        self.last_improved_step = 0
        self.neigh_attempts = [0.0] * len(NEIGHBORHOODS)
        self.neigh_success = [0.0] * len(NEIGHBORHOODS)
        return self._make_state()

    # this is what represents the GVNS Environment , i call here the b_vnd function with the selected neighborhood (the action chosen by the policy)
    # returns a reward and the new state
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        nb = NEIGHBORHOODS[int(action)]

        F_before = float(self.eval['F'])
        t0 = time.time()

        # run one BVND pass restricted to chosen neighborhood
        cand, improved, F_cand, eval_cand = b_vnd(
            self.inst, self.current,
            neighborhoods=[nb],
            w1=self.w1, w2=self.w2, M=self.M,
            max_total_iters=self.b_vnd_max_iters
        )
        elapsed = float(time.time() - t0)

        # record attempt
        self.neigh_attempts[action] += 1.0
        if improved:
            self.current = cand
            self.eval = eval_cand
            self.neigh_success[action] += 1.0

        F_after = float(self.eval['F'])
        # normalized improvement reward
        reward = (F_before - F_after) / (1.0 + abs(F_before))
        # small penalty if no improvement (keeps signal negative for wasted attempts)
        if not improved:
            reward += -0.01

        # time penalty (for faster improvements)
        reward -= self.time_penalty_coef * elapsed

        self.step_count += 1
        # termination conditions
        done = False
        if (time.time() - self.start_time) > self.Tmax:
            done = True
        if self.step_count >= self.max_steps:
            done = True

        info = {
            "neighborhood": nb,
            "improved": bool(improved),
            "F_before": F_before,
            "F_after": F_after,
            "elapsed": elapsed,
            "step": self.step_count
        }

        return self._make_state(), float(reward), bool(done), info

    def render(self, mode: str = "human"):
        print(f"step={self.step_count} F={self.eval['F']:.3f} attempts={self.neigh_attempts} success={self.neigh_success}")

    def close(self):
        pass