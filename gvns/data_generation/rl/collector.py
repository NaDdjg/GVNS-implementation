import os
import numpy as np

from typing import Dict, Any, List
from gvns.data_generation.rl.dataset import OfflineDataset


class GVNSCollector:
    """
    Collects transitions for PPO training:
    (state, action, reward, next_state, done)
    """

    def __init__(self, feature_names: List[str], save_path: str = None):
        self.save_path = save_path
        self.dataset = OfflineDataset(feature_names)
        self.reset()

    def reset(self):
        self._last_state = None
        self._last_action = None
        self._step_id = 0
        self.meta: List[Dict[str, Any]] = []

    # =========================================================
    # STEP HANDLING
    # =========================================================
    def record_decision(self, state: Dict[str, Any], action: int):
        """Called BEFORE LS"""
        self._last_state = state
        self._last_action = action

        if action < 0:
            print(f"[Warning] invalid action {action}")

    def record_outcome(
        self,
        next_state: Dict[str, Any],
        reward: float,
        done: bool,
        meta: Dict[str, Any] = None
    ):
        if self._last_state is None:
            return
    
        self.meta.append(meta if meta is not None else {})

        self.dataset.add_transition(
            state=self._last_state,
            action=self._last_action,
            reward=reward,
            next_state=next_state,
            done=done,
            meta=meta
        )

        self._last_state = None
        self._last_action = None
        self._step_id += 1

    def size(self):
        return len(self.dataset)

    # =========================================================
    # SAVE
    # =========================================================
    def save(self):
        if self.save_path is None:
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        self.dataset.save(self.save_path)