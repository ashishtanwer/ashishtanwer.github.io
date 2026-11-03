const n=`# AlphaZero on Connect Four, as the RL Explainer trains it on the CPU: the game on two bitboards, a small residual
# policy-value network, PUCT tree search guided by it, and the loss that pulls the network towards the search's visit
# counts and the games' results. scripts/rl/train_alphazero.py runs self-play, training and evaluation with these
# blocks; the check replays the recorded searches with the recorded weights.
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    rows: int = 6
    cols: int = 7
    channels: int = 48
    blocks: int = 3  # residual blocks
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0  # root noise in self-play only
    noise_fraction: float = 0.25
    lr: float = 1e-3  # Adam
    weight_decay: float = 1e-4


# %% [board] Connect Four on two bitboards
H1 = 7  # bits per column: six cells and one empty sentinel bit on top
BOTTOM = sum(1 << (c * H1) for c in range(7))


def legal(mask):
    """Columns whose top cell (bit 5 of the column) is still empty."""
    return [c for c in range(7) if not mask & (1 << (5 + c * H1))]


def play(state, c):
    """state = (stones of the player to move, all stones, moves so far). Dropping a piece in column c sets the lowest
    empty bit of that column (mask + the column's bottom bit carries up to it); then the other player is to move."""
    pos, mask, moves = state
    new_mask = mask | (mask + (1 << (c * H1)))
    return pos ^ mask, new_mask, moves + 1  # the opponent's stones become 'the player to move'


def four(bits):
    """Four in a row along any direction: vertical (shift 1), horizontal (7), and the two diagonals (6, 8)."""
    for shift in (1, H1, H1 - 1, H1 + 1):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


def outcome(state):
    """+1/−1/0 for the player to move if the game is over (the last mover won, so −1; a full board is 0), else None."""
    pos, mask, moves = state
    if four(pos ^ mask):  # the stones of the player who just moved
        return -1.0
    if moves == 42:
        return 0.0
    return None


EMPTY = (0, 0, 0)
_CELLS = np.array([c * H1 + r for r in range(6) for c in range(7)])  # row-major, bottom row first


def planes(state):
    """Network input (3, 6, 7): the mover's stones, the opponent's, and a plane of 1s when the mover went first."""
    pos, mask, moves = state
    def grid(bits):
        return np.unpackbits(np.frombuffer(bits.to_bytes(8, 'little'), np.uint8), bitorder='little')[_CELLS]
    first = np.full(42, 1.0 if moves % 2 == 0 else 0.0)
    return np.stack([grid(pos), grid(pos ^ mask), first]).reshape(3, 6, 7).astype(np.float32)


# %% [network] The policy-value network
class Residual(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Conv2d(ch, ch, 3, padding=1)
        self.b = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.b(F.relu(self.a(x))))


class PolicyValueNet(nn.Module):
    """Board (3, 6, 7) → seven move logits (the prior P) and a value v in [−1, 1], both for the player to move."""

    def __init__(self, c: Config):
        super().__init__()
        self.stem = nn.Conv2d(3, c.channels, 3, padding=1)
        self.tower = nn.Sequential(*[Residual(c.channels) for _ in range(c.blocks)])
        self.policy_conv = nn.Conv2d(c.channels, 2, 1)
        self.policy = nn.Linear(2 * 42, 7)
        self.value_conv = nn.Conv2d(c.channels, 1, 1)
        self.value_hidden = nn.Linear(42, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):  # (batch, 3, 6, 7)
        h = self.tower(F.relu(self.stem(x)))
        logits = self.policy(F.relu(self.policy_conv(h)).flatten(1))  # (batch, 7)
        v = torch.tanh(self.value(F.relu(self.value_hidden(F.relu(self.value_conv(h)).flatten(1)))))
        return logits, v.squeeze(1)


@torch.no_grad()
def evaluate(net, states):
    """Priors over the legal moves (illegal columns get 0) and values, for a list of states, in one batch."""
    logits, v = net(torch.from_numpy(np.stack([planes(s) for s in states])))
    priors = []
    for s, row in zip(states, logits):
        ok = legal(s[1])
        p = torch.softmax(row[ok], 0).tolist()
        priors.append(dict(zip(ok, p)))
    return priors, v.tolist()


# %% [puct] Choosing a move inside the tree
class Node:
    __slots__ = ('state', 'prior', 'children', 'N', 'W', 'terminal')

    def __init__(self, state, prior):
        self.state, self.prior, self.children = state, prior, {}
        self.N, self.W = 0, 0.0  # visits, and the summed value for the player who moved into this node
        self.terminal = outcome(state)

    def Q(self):
        return self.W / self.N if self.N else 0.0


def puct(parent, child, c_puct):
    """U = Q(s, a) + c · P(s, a) · √N(s) / (1 + N(s, a)): the value found so far plus a bonus for moves the prior
    likes and the search has not tried much."""
    return child.Q() + c_puct * child.prior * math.sqrt(parent.N) / (1 + child.N)


def select_leaf(root, c_puct):
    """Walk down by the best PUCT score to a node that is not expanded yet (or a finished game)."""
    path = [root]
    node = root
    while node.children and node.terminal is None:
        node = max(node.children.values(), key=lambda ch: puct(path[-1], ch, c_puct))
        path.append(node)
    return path


def expand(node, priors):
    for move, p in priors.items():
        node.children[move] = Node(play(node.state, move), p)


def backup(path, v):
    """v is the value for the player to move at the leaf; every node stores it for the player who moved into it."""
    for node in reversed(path):
        node.W -= v
        node.N += 1
        v = -v


# %% [mcts] The search
def add_noise(root, c: Config, rng):
    noise = rng.dirichlet([c.dirichlet_alpha] * len(root.children))
    for child, n in zip(root.children.values(), noise):
        child.prior = (1 - c.noise_fraction) * child.prior + c.noise_fraction * n


def mcts(net, state, sims, c: Config, rng=None, on_sim=None):
    """Expand the root with the network, then run \`sims\` simulations: select a leaf, score it with the network (or the
    game's result), expand it, back the value up. Returns the root, whose children's visit counts are the policy."""
    root = Node(state, 1.0)
    (priors,), (v,) = evaluate(net, [state])
    expand(root, priors)
    root.N = 1
    root.W = -v
    if rng is not None:
        add_noise(root, c, rng)
    for i in range(sims):
        path = select_leaf(root, c.c_puct)
        leaf = path[-1]
        if leaf.terminal is not None:
            value = leaf.terminal
        else:
            (priors,), (value,) = evaluate(net, [leaf.state])
            expand(leaf, priors)
        backup(path, value)
        if on_sim:
            on_sim(i + 1, root)
    return root


def visit_policy(root, temperature=1.0):
    """π(a) ∝ N(a)^(1/τ) over the root's moves; τ → 0 is the most-visited move."""
    moves = sorted(root.children)
    n = np.array([root.children[m].N for m in moves], float)
    if temperature == 0:
        p = (n == n.max()).astype(float)
    else:
        p = n ** (1 / temperature)
    pi = np.zeros(7)
    pi[moves] = p / p.sum()
    return pi


# %% [train-step] One training step
def train_step(net, optimizer, x, pi, z):
    """Loss = (z − v)² − Σ π · log p (+ weight decay in the optimiser): the value head learns the game's result, the
    policy head the search's visit distribution."""
    logits, v = net(x)
    value_loss = F.mse_loss(v, z)
    policy_loss = -(pi * F.log_softmax(logits, 1)).sum(1).mean()
    loss = value_loss + policy_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return value_loss.detach(), policy_loss.detach()
`;export{n as default};
