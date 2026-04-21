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
        self.normalized = False

        # main storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # metadata storage 
        self.instance_ids = []
        self.instance_sizes = []
        self.run_ids = []
        self.step_ids = []
        self.neigh_ids = []
        self.F_before = []
        self.F_after = []
        self.delta = []
        self.elapsed = []

    # =========================================================
    # State processing
    # =========================================================
    def _dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dict → ordered numpy vector.
        Missing features are filled with 0.0.
        """
        vec = np.array(
            [float(state_dict.get(f, 0.0)) for f in self.feature_names],
            dtype=np.float32
        )

        # sanitize
        vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)

        return vec

    # =========================================================
    # Add transition
    # =========================================================
    def add_transition(
        self,
        state: Dict[str, Any],
        action: int,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
        meta: Dict[str, Any] = None
    ):
        """
        Add one transition to dataset.
        """
        meta = meta or {}

        s_vec = self._dict_to_vector(state)
        ns_vec = self._dict_to_vector(next_state)

        # debug check 
        if np.allclose(s_vec, ns_vec):
            print("[Warning] state == next_state (possible bug)")

        # sanity checks 
        assert s_vec.shape[0] == self.feature_dim
        assert ns_vec.shape[0] == self.feature_dim

        self.states.append(s_vec)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(ns_vec)
        self.dones.append(bool(done))

        # metadata
        self.instance_ids.append(meta.get("instance_id", "unknown"))
        self.instance_sizes.append(meta.get("n_customers", 0))
        self.run_ids.append(meta.get("run_id", 0))
        self.step_ids.append(meta.get("step_id", 0))
        self.neigh_ids.append(meta.get("neigh_id", -1))
        self.F_before.append(meta.get("F_before", 0.0))
        self.F_after.append(meta.get("F_after", 0.0))
        self.delta.append(meta.get("delta", 0.0))
        self.elapsed.append(meta.get("elapsed", 0.0))

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
            "meta": {
                "instance_ids": np.array(self.instance_ids),
                "instance_sizes": np.array(self.instance_sizes),
                "run_ids": np.array(self.run_ids),
                "step_ids": np.array(self.step_ids),
                "neigh_ids": np.array(self.neigh_ids),
                "F_before": np.array(self.F_before),
                "F_after": np.array(self.F_after),
                "delta": np.array(self.delta),
                "elapsed": np.array(self.elapsed),
            }
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
            "normalized": self.normalized,
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
        assert payload["feature_names"] == obj.feature_names, \
            "Feature order mismatch!"
        
        obj.normalized = payload.get("normalized", False)
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

    @staticmethod
    def merge(datasets: List["OfflineDataset"]) -> "OfflineDataset":
        """
        Merge multiple datasets into one.
        """
        assert datasets, "No datasets to merge"

        base = datasets[0]
        merged = OfflineDataset(base.feature_names)

        # copy normalization flag
        merged.normalized = base.normalized

        for ds in datasets:
            assert ds.feature_names == base.feature_names, \
                "Feature mismatch between datasets"

            merged.states.extend(ds.states)
            merged.actions.extend(ds.actions)
            merged.rewards.extend(ds.rewards)
            merged.next_states.extend(ds.next_states)
            merged.dones.extend(ds.dones)

            if hasattr(ds, "instance_ids"):
                merged.instance_ids.extend(ds.instance_ids)
                merged.instance_sizes.extend(ds.instance_sizes)
                merged.run_ids.extend(ds.run_ids)
                merged.step_ids.extend(ds.step_ids)
                merged.neigh_ids.extend(ds.neigh_ids)
                merged.F_before.extend(ds.F_before)
                merged.F_after.extend(ds.F_after)
                merged.delta.extend(ds.delta)
                merged.elapsed.extend(ds.elapsed)

        return merged