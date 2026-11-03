const e=`# TD(0) and Monte Carlo prediction, then SARSA and Q-learning control, on CliffWalking-v1, as the RL Explainer runs
# them: a table of values (V for 48 states, or Q for 48 × 4 state–action pairs) nudged towards a target after each
# step or each episode. scripts/rl/record_td_learning.py runs these functions on NumPy arrays; they work the same on
# torch tensors, which is how the check replays the recorded runs from their seeds.
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    states: int = 48  # 4 × 12 cells, row by row; the start is 36, the goal 47, the cliff 37–46
    actions: int = 4  # up, right, down, left
    gamma: float = 1.0  # episodes always end, so no discount
    epsilon: float = 0.1  # how often an action is random
    alpha_td: float = 0.2  # TD(0)'s step size, the best of the sweep
    alpha_mc: float = 0.02  # Monte Carlo's
    alpha_control: float = 0.5  # SARSA's and Q-learning's


# %% [policy] The fixed policy and one episode
def route(s):
    """The route next to the cliff: up from the start, right along row 2, down at the end (and down from above)."""
    row, col = divmod(s, 12)
    return 0 if row == 3 else 1 if row == 2 and col < 11 else 2


def play(env, rng, c: Config):
    """One episode of ε-greedy around the route, in the real environment: a list of (s, a, r, s′, done)."""
    s, _ = env.reset()
    steps = []
    while True:
        a = route(s) if rng.random() >= c.epsilon else int(rng.integers(c.actions))
        s2, r, terminated, truncated, _ = env.step(a)
        steps.append((int(s), a, float(r), int(s2), bool(terminated)))
        s = s2
        if terminated or truncated:
            return steps


# %% [td-update] TD(0): learn from the next state's estimate
def td_update(V, s, r, s2, done, alpha, gamma):
    """V(s) ← V(s) + α · (r + γ·V(s′) − V(s)), right after the step. Returns the target."""
    target = r + (0.0 if done else gamma * V[s2])
    V[s] += alpha * (target - V[s])
    return target


# %% [mc-update] Monte Carlo: learn from the whole return
def returns(episode, gamma):
    """G_t = r_{t+1} + γ·r_{t+2} + … to the end of the episode, for every step t."""
    G, out = 0.0, []
    for s, a, r, s2, done in reversed(episode):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


def mc_update(V, episode, alpha, gamma):
    """After the episode: V(s_t) ← V(s_t) + α · (G_t − V(s_t)) for every visit, in order. Returns the targets."""
    G = returns(episode, gamma)
    for t, (s, *_) in enumerate(episode):
        V[s] += alpha * (G[t] - V[s])
    return G


# %% [epsilon-greedy] Acting from Q
def act(Q, s, epsilon, rng):
    """A random action with probability ε, otherwise the best by Q (ties broken at random)."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    row = np.asarray(Q[s])
    best = np.flatnonzero(row == row.max())
    return int(best[0] if len(best) == 1 else rng.choice(best))


# %% [sarsa] SARSA: the action it will really take next
def sarsa_update(Q, s, a, r, s2, a2, done, alpha, gamma):
    """Q(s, a) ← Q(s, a) + α · (r + γ·Q(s′, a′) − Q(s, a)), a′ chosen ε-greedily before the update."""
    target = r + (0.0 if done else gamma * Q[s2, a2])
    Q[s, a] += alpha * (target - Q[s, a])
    return target


# %% [qlearning] Q-learning: the best action next, whatever it does
def q_learning_update(Q, s, a, r, s2, done, alpha, gamma):
    """Q(s, a) ← Q(s, a) + α · (r + γ·max_a′ Q(s′, a′) − Q(s, a))."""
    target = r + (0.0 if done else gamma * Q[s2].max())
    Q[s, a] += alpha * (target - Q[s, a])
    return target


def control_episode(env, Q, method, rng, c: Config):
    """One learning episode of SARSA or Q-learning. Returns the sum of rewards, the falls, the steps and the cells
    left (each step's s)."""
    s, _ = env.reset()
    a = act(Q, s, c.epsilon, rng)
    total, falls, cells = 0.0, 0, []
    while True:
        s2, r, terminated, truncated, _ = env.step(a)
        total += r
        falls += r == -100
        cells.append(int(s))
        if method == 'sarsa':
            a2 = act(Q, s2, c.epsilon, rng)
            sarsa_update(Q, s, a, r, s2, a2, terminated, c.alpha_control, c.gamma)
        else:
            q_learning_update(Q, s, a, r, s2, terminated, c.alpha_control, c.gamma)
        s = s2
        if terminated or truncated:
            return total, falls, cells
        a = a2 if method == 'sarsa' else act(Q, s, c.epsilon, rng)


def as_table(states, actions=None):
    """A float64 table of zeros (torch, for the check's replays)."""
    return torch.zeros(states if actions is None else (states, actions), dtype=torch.float64)
`;export{e as default};
