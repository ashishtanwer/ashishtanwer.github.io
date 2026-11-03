const e=`# Value iteration on FrozenLake 8×8, as the RL Explainer computes it: the environment's transition table as tensors,
# the Bellman optimality backup over every state at once, sweeps until the residual is tiny, and the greedy policy read
# off the result. The check compares these blocks with the recorded sweeps of both maps.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    states: int = 64  # 8 × 8 cells, row by row
    actions: int = 4  # left, down, right, up
    gamma: float = 0.99
    tolerance: float = 1e-8  # stop when no value moves by more than this in a sweep


# %% [transitions] The transition table
def transition_tensors(P, c: Config):
    """P[s][a] = [(p, s′, r, done), …], Gymnasium's own table (env.unwrapped.P). Returns T[s, a, s′] = p(s′ | s, a),
    R[s, a], the expected reward, and D[s, a, s′] = 1 where that move ends the episode."""
    T = torch.zeros(c.states, c.actions, c.states, dtype=torch.float64)
    R = torch.zeros(c.states, c.actions, dtype=torch.float64)
    D = torch.zeros(c.states, c.actions, c.states, dtype=torch.float64)
    for s in range(c.states):
        for a in range(c.actions):
            for p, s2, r, done in P[s][a]:
                T[s, a, s2] += p
                R[s, a] += p * r
                D[s, a, s2] = float(done)
    return T, R, D


# %% [bellman-backup] One Bellman backup
def backup(V, T, R, D, gamma):
    """Q(s, a) = Σ_s′ p(s′ | s, a) · (r + γ · V(s′)), with no future once the episode has ended; for every s and a."""
    return R + gamma * torch.einsum('sat,sat,t->sa', T, 1 - D, V)


# %% [value-iteration] Sweeps until nothing moves
def value_iteration(T, R, D, c: Config):
    """V₀ = 0; V_k(s) = max_a Q_k(s, a), every state from V_{k−1} (a synchronous sweep). Returns every V_k and the
    residual max_s |V_k − V_{k−1}| of each sweep, which shrinks by about γ per sweep."""
    V = torch.zeros(c.states, dtype=torch.float64)
    values, residuals = [V], []
    while True:
        V_new = backup(V, T, R, D, c.gamma).max(1).values
        residuals.append((V_new - V).abs().max())
        values.append(V_new)
        V = V_new
        if residuals[-1] < c.tolerance:
            return torch.stack(values), torch.stack(residuals)


# %% [greedy-policy] The greedy policy and how often it reaches the goal
def greedy_policy(V, T, R, D, gamma):
    """π(s) = argmax_a Q(s, a): one more backup, then the best action of every state."""
    return backup(V, T, R, D, gamma).argmax(1)


def success_probability(T, policy, goal, end, horizon=None):
    """P(reach G) from every state when following π: p = [s is G] + [s not G or H] · P_π p, solved exactly, or
    iterated \`horizon\` times for the chance within a time limit."""
    P = T[torch.arange(len(policy)), policy] * (~end).double()[:, None]  # no moves out of G or a hole
    g = goal.double()
    if horizon is None:
        return torch.linalg.solve(torch.eye(len(policy), dtype=torch.float64) - P, g)
    p = g.clone()
    for _ in range(horizon):
        p = g + P @ p
    return p
`;export{e as default};
