# gvns/agents/ppo_agent.py

import numpy as np
import pickle
from stable_baselines3 import PPO
from .base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, model_path=None, vecnorm_path=None, device="cpu"):
        self.model = None
        self.obs_rms = None
        self.uses_vecnorm = False
        self.device = device

        if model_path is not None:
            self._load_model(model_path)
        if vecnorm_path is not None:
            self._load_vecnorm(vecnorm_path)

    def _load_model(self, path):
        self.model = PPO.load(path, device=self.device)

    def _load_vecnorm(self, path):
        try:
            with open(path, "rb") as f:
                vecnorm = pickle.load(f)
            self.obs_rms = vecnorm.obs_rms
            self.uses_vecnorm = True
        except Exception:
            self.obs_rms = None
            self.uses_vecnorm = False

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.uses_vecnorm or self.obs_rms is None:
            return obs.astype(np.float32)
        o = np.array(obs, dtype=np.float32)
        if o.ndim == 1:
            o = o.reshape(1, -1)
        mean = getattr(self.obs_rms, "mean", None)
        var = getattr(self.obs_rms, "var", None)
        if mean is None or var is None:
            return o
        normed = (o - mean) / (np.sqrt(var + 1e-8))
        return np.clip(normed, -10.0, 10.0).astype(np.float32)

    # agent API
    def select_action(self, state: np.ndarray, deterministic: bool = True) -> int:
        obs = self._normalize_obs(state)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(np.asarray(action).reshape(-1)[0])

    def update(self, *args, **kwargs):
        """PPO is trained offline so no online update here."""
        pass

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path + "_ppo.zip")
        if self.obs_rms is not None:
            with open(path + "_vecnorm.pkl", "wb") as f:
                pickle.dump({"obs_rms": self.obs_rms}, f)

    def load(self, path: str):
        self._load_model(path + "_ppo.zip")
        try:
            with open(path + "_vecnorm.pkl", "rb") as f:
                data = pickle.load(f)
            self.obs_rms = data["obs_rms"]
            self.uses_vecnorm = True
        except FileNotFoundError:
            self.obs_rms = None
            self.uses_vecnorm = False
