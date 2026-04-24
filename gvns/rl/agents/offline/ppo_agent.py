import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gvns.rl.agents.base_agent import BaseAgent

# =========================================================
# Actor-Critic Network
# =========================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # policy head
        self.policy_head = nn.Linear(hidden_dim, n_actions)

        # value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

# =========================================================
# PPO Agent
# =========================================================
class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        feature_names: Optional[List[str]] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        k_epochs: int = 4,
        batch_size: int = 64,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.feature_names = list(feature_names) if feature_names is not None else None

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # =====================================================
    # Helpers
    # =====================================================
    def _state_to_vector(self, state: Any) -> np.ndarray:
        """
        Convert a state dict or array-like to a flat float32 vector
        in the correct feature order.
        """
        if isinstance(state, dict):
            if self.feature_names is None:
                raise ValueError(
                    "PPOAgent received a dict state, but feature_names were not provided."
                )
            vec = np.array([float(state.get(k, 0.0)) for k in self.feature_names], dtype=np.float32)
        else:
            vec = np.asarray(state, dtype=np.float32).reshape(-1)

        if vec.shape[0] != self.state_dim:
            raise ValueError(
                f"State dimension mismatch: got {vec.shape[0]}, expected {self.state_dim}"
            )

        vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
        return vec

    def _unpack_dataset(self, dataset: Any) -> Dict[str, np.ndarray]:
        """
        Accept either:
        - OfflineDataset (has to_numpy())
        - dict of arrays
        """
        if hasattr(dataset, "to_numpy"):
            data = dataset.to_numpy()
        elif isinstance(dataset, dict):
            data = dataset
        else:
            raise TypeError(
                "dataset must be an OfflineDataset or a dict with keys "
                "states/actions/rewards/dones"
            )

        required = ["states", "actions", "rewards", "dones"]
        for key in required:
            if key not in data:
                raise KeyError(f"Missing key in dataset: {key}")

        return data

    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute discounted returns.
        If done flags exist, returns reset at episode boundaries.
        """
        returns = torch.zeros_like(rewards)
        G = 0.0

        for t in reversed(range(len(rewards))):
            if bool(dones[t].item()):
                G = 0.0
            G = rewards[t].item() + self.gamma * G
            returns[t] = G

        return returns
    
    # =====================================================
    # Action selection (used inside GVNS)
    # =====================================================
    def select_action(self, state: Any) -> int:
        """
        Returns an action index sampled from the current policy.
        """
        state_vec = self._state_to_vector(state)
        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
        self.model.train()

        return int(action)

    # =====================================================
    # Offline PPO-style training
    # =====================================================
    def train_on_dataset(
        self,
        dataset: Any,
        epochs: int = 4,
        shuffle: bool = True,
        grad_clip: float = 0.5,
        normalize_returns: bool = True,
        normalize_advantages: bool = True,
    ) -> Dict[str, float]:
        """
        Offline PPO-style update on logged trajectories.

        Important:
        - Best used on one run-file at a time, or on data where episode
          boundaries are preserved with done flags.
        """
        data = self._unpack_dataset(dataset)

        states = torch.as_tensor(data["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(data["actions"], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(data["rewards"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(data["dones"], dtype=torch.float32, device=self.device)

        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(
                f"Dataset states shape mismatch: got {tuple(states.shape)}, "
                f"expected [N, {self.state_dim}]"
            )

        returns = self._compute_returns(rewards, dones)

        if normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        self.model.train()

        with torch.no_grad():
            old_logits, old_values = self.model(states)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions)

        advantages = (returns - old_values.squeeze(-1)).detach()
        if normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]
        indices = np.arange(n)

        history = {
            "loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
        }

        n_steps = 0

        for _ in range(epochs):
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]

                logits, values = self.model(b_states)
                dist = torch.distributions.Categorical(logits=logits)

                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - b_old_log_probs)

                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(-1), b_returns)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                history["loss"] += loss.item()
                history["actor_loss"] += actor_loss.item()
                history["critic_loss"] += critic_loss.item()
                history["entropy"] += entropy.item()
                n_steps += 1

        if n_steps > 0:
            for k in history:
                history[k] /= n_steps

        return history

    # =====================================================
    # BaseAgent interface compatibility
    # =====================================================
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """
        Offline training is handled by train_on_dataset().
        This method stays for interface compatibility.
        """
        return

    # =====================================================
    # Save / Load
    # =====================================================
    def save(self, path: str):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "feature_names": self.feature_names,
            "gamma": self.gamma,
            "clip_eps": self.clip_eps,
            "k_epochs": self.k_epochs,
            "batch_size": self.batch_size,
        }
        torch.save(payload, path)

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device)

        if isinstance(payload, dict) and "model_state_dict" in payload:
            self.model.load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                self.optimizer.load_state_dict(payload["optimizer_state_dict"])

            self.state_dim = payload.get("state_dim", self.state_dim)
            self.n_actions = payload.get("n_actions", self.n_actions)
            self.feature_names = payload.get("feature_names", self.feature_names)
            self.gamma = payload.get("gamma", self.gamma)
            self.clip_eps = payload.get("clip_eps", self.clip_eps)
            self.k_epochs = payload.get("k_epochs", self.k_epochs)
            self.batch_size = payload.get("batch_size", self.batch_size)
        else:
            self.model.load_state_dict(payload)

        self.model.eval()