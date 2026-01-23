import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Tuple, Any

from .base_agent import BaseAgent
from gvns.utils import RunningMeanStd

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# MLP used for the DQN
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128,64), n_actions=4):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_actions))  # output Q-values (one per action)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# agent
class DQNAgent(BaseAgent):
    def __init__(self,
                 input_dim: int,
                 n_actions: int = 4,
                 hidden_dims=(128,64),
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 buffer_size: int = 50_000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 start_train_steps: int = 1000,
                 min_buffer_size: int = 1000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay_steps: int = 50_000,
                 device: str = "cpu",
                 seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.start_train_steps = start_train_steps
        self.min_buffer_size = min_buffer_size

        # networks
        self.q_net = QNetwork(input_dim, hidden_dims, n_actions).to(self.device)
        self.target_net = QNetwork(input_dim, hidden_dims, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # eps schedule
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.eps_decay = (eps_start - eps_end) / max(1, eps_decay_steps)

        # running stats for normalization (state dim)
        self.rms = RunningMeanStd(input_dim)

        # counters
        self.total_steps = 0
        self.learn_steps = 0

    # agent API
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        obs: raw observation vector (1d numpy)
        returns: action int
        """
        # update running mean/std online (small stabilizer)
        self.rms.update(obs)
        s = self.rms.normalize(obs)
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)  # shape (1, dim)

        if deterministic:
            with torch.no_grad():
                qvals = self.q_net(s_t)
                action = int(torch.argmax(qvals, dim=1).item())
            return action

        # epsilon-greedy
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            qvals = self.q_net(s_t)
            return int(torch.argmax(qvals, dim=1).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # store raw numpy arrays
        self.buffer.push(state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), bool(done))
        self.total_steps += 1
        # decay epsilon
        if self.eps > self.eps_end:
            self.eps = max(self.eps_end, self.eps - self.eps_decay)

    def train_step(self):
        if len(self.buffer) < self.min_buffer_size:
            return 0.0  # not enough data

        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.vstack(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # normalize using stored running stats
        states = torch.tensor(self.rms.normalize(states.cpu().numpy()), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(self.rms.normalize(next_states.cpu().numpy()), dtype=torch.float32, device=self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)  # shape (B,1)
        # compute target: r + gamma * max_a' Q_target(s', a') * (1-done)
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + (1.0 - dones) * self.gamma * q_next_max

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1

        # periodic target update
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def update(self, *args, **kwargs):
        # uses store_transition + train_step called by trainer script
        raise NotImplementedError("Use store_transition() and train_step() directly for DQNAgent")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_state_dict': self.q_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data['q_state_dict'])
        self.target_net.load_state_dict(data['target_state_dict'])
        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
