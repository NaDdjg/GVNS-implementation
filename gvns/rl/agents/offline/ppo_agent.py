import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from typing import Any
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
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        k_epochs: int = 4,
        batch_size: int = 64,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.memory = []  # used if online 

    # =====================================================
    # Action selection (used inside GVNS)
    # =====================================================
    def select_action(self, state: Any) -> int:
        """
        state = feature vector (dict or np.array)
        returns action index
        """
        if isinstance(state, dict):
            state = np.array(list(state.values()), dtype=np.float32)

        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(state_t)
            probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()

        return action

    # =====================================================
    # PPO update (offline training)
    # =====================================================
    def train_on_dataset(self, dataset):
        """
        dataset must provide:
        states, actions, rewards, next_states, dones
        """

        states = torch.tensor(dataset["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(dataset["actions"], dtype=torch.long).to(self.device)
        rewards = torch.tensor(dataset["rewards"], dtype=torch.float32).to(self.device)
        dones = torch.tensor(dataset["dones"], dtype=torch.float32).to(self.device)

        # compute returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        with torch.no_grad():
            logits, values = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            old_log_probs = dist.log_prob(actions)

        # PPO training loop
        for _ in range(self.k_epochs):
            for i in range(0, len(states), self.batch_size):
                batch_slice = slice(i, i + self.batch_size)

                b_states = states[batch_slice]
                b_actions = actions[batch_slice]
                b_returns = returns[batch_slice]
                b_old_log_probs = old_log_probs[batch_slice]

                logits, values = self.model(b_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)

                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - b_old_log_probs)

                advantages = b_returns - values.squeeze()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values.squeeze(), b_returns)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # =====================================================
    # Save / Load
    # =====================================================
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()