const e=`# DQN on CartPole-v1, as the RL Explainer trains it on the CPU: a small Q-network, a replay buffer, ε-greedy actions,
# and a target network for the bootstrapped target. scripts/rl/train_dqn.py trains with these blocks; the check
# replays the recorded update and recorded Q-values against them.
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    obs_dim: int = 4  # cart position, cart velocity, pole angle, pole angular velocity
    actions: int = 2  # push the cart left (0) or right (1)
    hidden: int = 128
    gamma: float = 0.99
    lr: float = 2.5e-4  # Adam


# %% [q-network] The Q-network
class QNetwork(nn.Module):
    """Q(s, ·): one value per action, the return the network expects from pushing that way and then acting well."""

    def __init__(self, c: Config):
        super().__init__()
        self.fc1 = nn.Linear(c.obs_dim, c.hidden)
        self.fc2 = nn.Linear(c.hidden, c.hidden)
        self.out = nn.Linear(c.hidden, c.actions)

    def forward(self, s):  # (batch, 4) → (batch, 2)
        return self.out(F.relu(self.fc2(F.relu(self.fc1(s)))))


# %% [act] ε-greedy actions
def act(q, s, epsilon, rng):
    """With probability ε a random push, otherwise the action with the larger Q-value."""
    if rng.random() < epsilon:
        return int(rng.integers(2))
    with torch.no_grad():
        return int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())


# %% [replay-buffer] The replay buffer
class ReplayBuffer:
    """A ring of the last \`capacity\` transitions (s, a, r, s′, terminated); \`step\` records when each was added."""

    def __init__(self, capacity, obs_dim):
        self.capacity, self.size, self.next = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)
        self.step = np.zeros(capacity, np.int64)

    def add(self, s, a, r, s2, terminated, step):
        i = self.next
        self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i], self.step[i] = s, a, r, s2, terminated, step
        self.next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def batch(self, idx):
        return {'s': torch.from_numpy(self.s[idx]), 'a': torch.from_numpy(self.a[idx]),
                'r': torch.from_numpy(self.r[idx]), 's2': torch.from_numpy(self.s2[idx]),
                'done': torch.from_numpy(self.done[idx])}

    def sample(self, n, rng):
        """n transitions drawn uniformly from everything stored: neighbours in time rarely meet in one batch."""
        return self.batch(rng.integers(0, self.size, n))

    def latest(self, n):
        """The n newest transitions, oldest first (the no-replay run learns only from these)."""
        return self.batch((self.next - n + np.arange(n)) % self.capacity)

    def snapshot(self, n, rng):
        idx = rng.choice(self.size, min(n, self.size), replace=False)
        return {'s': self.s[idx].copy(), 'a': self.a[idx].copy(), 'r': self.r[idx].copy(),
                'done': self.done[idx].copy(), 'step': self.step[idx].copy(), 'size': self.size}


# %% [td-target] The TD target
@torch.no_grad()
def td_target(target_net, r, s2, done, gamma):
    """y = r + γ · max_a′ Q_target(s′, a′), and y = r when the pole fell (a 500-step time-out still bootstraps)."""
    return r + gamma * (1 - done) * target_net(s2).max(1).values


# %% [train-step] One update
def train_step(q, target_net, batch, optimizer, c: Config):
    """Squared TD error on a minibatch, one Adam step. Returns the loss and the batch's mean max-Q before the step."""
    y = td_target(target_net, batch['r'], batch['s2'], batch['done'], c.gamma)
    qs = q(batch['s'])  # (batch, 2)
    q_sa = qs.gather(1, batch['a'][:, None]).squeeze(1)  # Q(s, a) of the action taken
    loss = F.mse_loss(q_sa, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach(), qs.detach().max(1).values.mean()
`;export{e as default};
