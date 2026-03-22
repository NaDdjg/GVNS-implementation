import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

from .base_agent import BaseAgent
from gvns.utils.rl_utils import RunningMeanStd, obs_dim

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))

# ================================
# Q Network: MLP used for the DQN
# ================================
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# ================================
# Replay buffer
# ================================
class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)
    
# ================================
# Agent
# ================================
class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        min_buffer_size: int = 300,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 300,
        device: str = "cpu",
        seed: int = 0
    ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        input_dim = obs_dim(n_actions) 

        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_size = min_buffer_size

        # networks
        self.q_net = QNetwork(input_dim, n_actions).to(self.device)
        self.target_net = QNetwork(input_dim, n_actions).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # eps schedule
        self.eps = eps_start
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

        # state normalization (state dim)
        self.rms = RunningMeanStd(input_dim)

        # counters
        self.total_steps = 0
        self.learn_steps = 0


    # Action selection
    def select_action(self, obs: np.ndarray) -> int:
        """
        obs: raw observation vector (1d numpy)
        returns: action int
        """
        # update running mean/std online (small stabilizer)
        state = obs.astype(np.float32)
        state = self.rms.normalize(state)

        state_t = torch.tensor(
            state, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0)  # shape (1, dim)

        # epsilon-greedy
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            qvals = self.q_net(state_t)

        return int(torch.argmax(qvals, dim=1).item())

    # Update: store transition 
    def update(self, state, action, reward, next_state, done):
        self.rms.update(state)
        self.rms.update(next_state)

        self.buffer.push(
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done)
        )

        self.total_steps += 1

        # epsilon decay
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        self.eps = self.eps_start - frac * (self.eps_start - self.eps_end)

        if len(self.buffer) < self.min_buffer_size:
            return

        # to double learning speed 
        for _ in range(2):
            self.train_step()

    # DQN training step
    def train_step(self):
        batch = self.buffer.sample(self.batch_size)

        states = np.vstack(batch.state)
        next_states = np.vstack(batch.next_state)

        states = self.rms.normalize(states)
        next_states = self.rms.normalize(next_states)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        actions = torch.tensor(batch.action, 
                               dtype=torch.int64, 
                               device=self.device).unsqueeze(1)
        
        rewards = torch.tensor(batch.reward, 
                               dtype=torch.float32, 
                               device=self.device).unsqueeze(1)
        
        dones = torch.tensor(batch.done, 
                             dtype=torch.float32, 
                             device=self.device).unsqueeze(1)
        
        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions) 

        # compute target: r + gamma * max_a' Q_target(s', a') * (1-done)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions)

            target = rewards + (1.0 - dones) * self.gamma * q_next

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1

        # periodic target update
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.learn_steps % 50 == 0:
            print("learn step", self.learn_steps)

        return loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)

        self.q_net.load_state_dict(data['q_net'])
        self.target_net.load_state_dict(data['target_net'])

        if 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
