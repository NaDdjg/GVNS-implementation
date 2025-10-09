"""
interface for all RL agents.
all agents implement:
  - select_action(state_or_key) -> int
  - update(state_or_key, action, reward, next_state_or_key, done) 
  - save(path), load(path)
"""

from typing import Any

class BaseAgent:
    def select_action(self, state: Any) -> int:
        """return an action index (0..n-1) for the provided state or state_key."""
        raise NotImplementedError

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        pass

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError
