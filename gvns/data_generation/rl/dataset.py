import numpy as np
import pickle
from typing import List, Dict, Any

class OfflineDataset:
    """
    Dataset for training PPO controller on GVNS.

    Stores transitions:
        (state, action, reward, next_state, done)

    Internally uses lists for flexibility,
    converts to numpy arrays when saving.
    """

    def __init__(self, feature_names: List[str]):
        """
        Args:
            feature_names: ordered list of features used to build state vectors
        """
        self.feature_names = feature_names
        self.feature_dim = len(feature_names)

        # storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    # =========================================================
    # State processing
    # =========================================================
    def _dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dict → ordered numpy vector.
        Missing features are filled with 0.0.
        """
        return np.array(
            [float(state_dict.get(f, 0.0)) for f in self.feature_names],
            dtype=np.float32
        )

    # =========================================================
    # Add transition
    # =========================================================
    def add_transition(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool
    ):
        """
        Add one transition to dataset.
        """

        s_vec = self._dict_to_vector(state)
        ns_vec = self._dict_to_vector(next_state)

        # sanity checks (VERY important)
        assert s_vec.shape[0] == self.feature_dim
        assert ns_vec.shape[0] == self.feature_dim

        self.states.append(s_vec)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(ns_vec)
        self.dones.append(bool(done))

    # =========================================================
    # Size
    # =========================================================
    def __len__(self):
        return len(self.states)

    # =========================================================
    # Convert to numpy
    # =========================================================
    def to_numpy(self):
        """
        Convert internal lists to numpy arrays.
        """
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "next_states": np.array(self.next_states, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.bool_),
        }

    # =========================================================
    # Save / Load
    # =========================================================
    def save(self, path: str):
        """
        Save dataset to disk.
        """
        data = self.to_numpy()

        payload = {
            "data": data,
            "feature_names": self.feature_names,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"[Dataset] Saved {len(self)} transitions → {path}")

    @staticmethod
    def load(path: str):
        """
        Load dataset from disk.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = OfflineDataset(feature_names=payload["feature_names"])

        data = payload["data"]

        obj.states = list(data["states"])
        obj.actions = list(data["actions"])
        obj.rewards = list(data["rewards"])
        obj.next_states = list(data["next_states"])
        obj.dones = list(data["dones"])

        print(f"[Dataset] Loaded {len(obj)} transitions from {path}")

        return obj

    # =========================================================
    # Debug / stats
    # =========================================================
    def describe(self):
        """
        Print basic dataset statistics.
        """
        n = len(self)

        if n == 0:
            print("[Dataset] Empty dataset.")
            return

        rewards = np.array(self.rewards)

        print("====== DATASET STATS ======")
        print(f"Transitions: {n}")
        print(f"Feature dim: {self.feature_dim}")
        print(f"Reward mean: {rewards.mean():.4f}")
        print(f"Reward std : {rewards.std():.4f}")
        print(f"Reward min : {rewards.min():.4f}")
        print(f"Reward max : {rewards.max():.4f}")

        unique, counts = np.unique(self.actions, return_counts=True)
        print("Action distribution:")
        for u, c in zip(unique, counts):
            print(f"  action {u}: {c} ({c/n:.2%})")

        done_ratio = np.mean(self.dones)
        print(f"Done ratio: {done_ratio:.4f}")
        print("===========================")