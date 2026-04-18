import os
import numpy as np
from typing import Dict, Any, List


class GVNSCollector:
    """
    Collects transitions for PPO training:
    (state, action, reward, next_state, done)
    """

    def __init__(self, action_dim: int, save_path: str = None):
        self.action_dim = action_dim
        self.save_path = save_path

        self.reset()

    def reset(self):
        self.states: List[Dict[str, Any]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[Dict[str, Any]] = []
        self.dones: List[int] = []

        # temp storage for current step
        self._last_state = None
        self._last_action = None

    # =========================================================
    # STEP HANDLING
    # =========================================================
    def record_decision(self, state: Dict[str, Any], action: int):
        """Called BEFORE LS"""
        self._last_state = state
        self._last_action = action

    def record_outcome(
        self,
        next_state: Dict[str, Any],
        reward: float,
        done: bool
    ):
        """Called AFTER LS"""
        if self._last_state is None:
            return

        self.states.append(self._last_state)
        self.actions.append(self._last_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(int(done))

        self._last_state = None
        self._last_action = None

    # =========================================================
    # SAVE
    # =========================================================
    def save(self):
        if self.save_path is None:
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        np.savez(
            self.save_path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            next_states=self.next_states,
            dones=self.dones,
        )

        print(f"[Collector] Saved dataset to {self.save_path}")