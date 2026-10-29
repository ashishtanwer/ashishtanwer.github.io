const n=`# TD-MPC2 (Hansen, Su & Wang 2024), nicklashansen/tdmpc2 dmcontrol/walker-run-1.pt, the agent as plain PyTorch.
# An encoder maps the 24 state numbers of DeepMind Control's walker to a 512-wide latent made of 64 softmax groups
# (SimNorm). A dynamics MLP predicts the next latent from a latent and an action; heads predict the reward (two-hot
# over 101 bins in symlog space), five Q values and a Gaussian policy prior. There is no decoder: the latent is
# trained only to predict rewards, values and its own next latent. To act, MPPI samples action sequences, rolls them
# through the dynamics, scores them with the rewards and a terminal Q, and refits a Gaussian to the best.
# Each "# %% [id]" section is one step of the TD-MPC2 tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    obs_dim: int = 24  # walker: 14 orientations, height, 9 velocities
    action_dim: int = 6  # hips, knees and ankles, in [-1, 1]
    enc_dim: int = 256
    mlp_dim: int = 512
    latent_dim: int = 512
    simnorm_dim: int = 8  # 64 groups of 8
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.0  # bins span [-10, 10] in symlog space
    vmax: float = 10.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    discount: float = 0.99  # walker's 500-step episodes: (500/5 − 1)/(500/5)
    horizon: int = 3
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24  # sequences proposed by the policy prior
    temperature: float = 0.5
    min_std: float = 0.05
    max_std: float = 2.0


class NormedLinear(nn.Module):
    """Linear → LayerNorm → activation (Mish unless given): the layer every TD-MPC2 MLP is made of."""

    def __init__(self, d_in: int, d_out: int, act: nn.Module | None = None):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.act = act if act is not None else nn.Mish()

    def forward(self, x):
        return self.act(self.ln(self.linear(x)))


def mlp(d_in: int, hidden: list[int], d_out: int, act: nn.Module | None = None) -> nn.Sequential:
    """Hidden NormedLinear layers, then a plain Linear (or a NormedLinear with \`act\`, as the encoder and dynamics)."""
    dims = [d_in, *hidden]
    layers = [NormedLinear(a, b) for a, b in zip(dims[:-1], dims[1:])]
    layers.append(NormedLinear(dims[-1], d_out, act) if act is not None else nn.Linear(dims[-1], d_out))
    return nn.Sequential(*layers)


# %% [simnorm] SimNorm: the latent as 64 softmax groups
class SimNorm(nn.Module):
    """Splits the vector into groups of 8 and puts each through a softmax, so every group is a point on a simplex:
    non-negative, summing to 1. It keeps the latent bounded however far the dynamics are rolled."""

    def __init__(self, c: Config):
        super().__init__()
        self.dim = c.simnorm_dim

    def forward(self, x):  # (..., 512)
        groups = x.view(*x.shape[:-1], -1, self.dim)  # (..., 64, 8)
        return F.softmax(groups, dim=-1).view(x.shape)


# %% [encoder] Encoder: state → latent
class Encoder(nn.Module):
    """24 state numbers → 256 (Mish) → 512 → SimNorm. This latent z is the only state TD-MPC2 plans in."""

    def __init__(self, c: Config):
        super().__init__()
        self.net = mlp(c.obs_dim, [c.enc_dim], c.latent_dim, act=SimNorm(c))

    def forward(self, obs):  # (batch, 24) → (batch, 512)
        return self.net(obs)


# %% [dynamics] Latent dynamics
class Dynamics(nn.Module):
    """z and an action → the next z, through two hidden layers of 512 and a SimNorm output. Trained to match the
    encoder's latent of the observation that really came next (the consistency loss), never to rebuild pixels."""

    def __init__(self, c: Config):
        super().__init__()
        self.net = mlp(c.latent_dim + c.action_dim, [c.mlp_dim, c.mlp_dim], c.latent_dim, act=SimNorm(c))

    def forward(self, z, a):  # (batch, 512), (batch, 6) → (batch, 512)
        return self.net(torch.cat([z, a], dim=-1))


# %% [reward] Reward head: two-hot over 101 bins
def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


def two_hot(x, c: Config):
    """A scalar → its two-hot target: symlog(x) lies between two neighbouring bins, which share the weight."""
    bin_size = (c.vmax - c.vmin) / (c.num_bins - 1)
    x = symlog(x).clamp(c.vmin, c.vmax)  # (batch,)
    position = (x - c.vmin) / bin_size
    lower = position.floor().long()
    upper_weight = position - lower
    target = torch.zeros(x.shape[0], c.num_bins)
    target.scatter_(1, lower[:, None], (1 - upper_weight)[:, None])
    target.scatter_(1, ((lower + 1) % c.num_bins)[:, None], upper_weight[:, None])
    return target


def two_hot_inv(logits, c: Config):
    """Logits over the bins → a scalar: the softmax-weighted mean bin, mapped back with symexp."""
    bins = torch.linspace(c.vmin, c.vmax, c.num_bins)
    return symexp((F.softmax(logits, dim=-1) * bins).sum(-1, keepdim=True))


class RewardHead(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.net = mlp(c.latent_dim + c.action_dim, [c.mlp_dim, c.mlp_dim], c.num_bins)

    def forward(self, z, a):  # → logits (batch, 101); the loss is cross-entropy with two_hot(real reward)
        return self.net(torch.cat([z, a], dim=-1))


# %% [q-ensemble] Five Q heads
class QEnsemble(nn.Module):
    """Five independent MLPs, each a two-hot distribution over the return of taking \`a\` in \`z\`. Planning and the
    policy average two of them chosen at random; the TD target takes the smaller of two (of the slow copies)."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = nn.ModuleList(mlp(c.latent_dim + c.action_dim, [c.mlp_dim, c.mlp_dim], c.num_bins)
                                   for _ in range(c.num_q))

    def forward(self, z, a):  # → (5, batch, 101)
        x = torch.cat([z, a], dim=-1)
        return torch.stack([head(x) for head in self.heads])

    def value(self, z, a, c: Config, pick: torch.Tensor, reduce: str = 'avg'):
        q = two_hot_inv(self(z, a)[pick], c)  # (2, batch, 1): the two chosen heads
        return q.mean(0) if reduce == 'avg' else q.min(0).values


# %% [policy] Policy prior
class PolicyPrior(nn.Module):
    """z → mean and log-std of a Gaussian over the 6 actions, squashed by tanh. It proposes 24 of the planner's
    sequences and, at the end of the horizon, the action whose Q closes the estimated return."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.net = mlp(c.latent_dim, [c.mlp_dim, c.mlp_dim], 2 * c.action_dim)

    def forward(self, z, eps):  # eps: standard normal noise (batch, 6)
        mean, log_std = self.net(z).chunk(2, dim=-1)
        log_std = self.c.log_std_min + 0.5 * (self.c.log_std_max - self.c.log_std_min) * (torch.tanh(log_std) + 1)
        return torch.tanh(mean + eps * log_std.exp()), torch.tanh(mean)  # sampled action, mean action


# %% [planner] MPPI planning in latent space
@torch.no_grad()
def estimate_value(m: 'TDMPC2', z, actions, c: Config, eps, pick):
    """The return of each sequence: predicted rewards along the dynamics, then a discounted Q at the end."""
    G, discount = 0, 1
    for t in range(c.horizon):
        G = G + discount * two_hot_inv(m.reward(z, actions[t]), c)
        z = m.dynamics(z, actions[t])
        discount = discount * c.discount
    a, _ = m.pi(z, eps)
    return G + discount * m.q.value(z, a, c, pick)  # (samples, 1)


@torch.no_grad()
def plan(m: 'TDMPC2', obs, c: Config, noise, prev_mean=None):
    """One planning step. \`noise\` holds the random draws (the policy's eps, the Gaussian samples, the Q-head
    choices), so the plan is a function of its inputs. Returns the executed action and the final mean."""
    z = m.encoder(obs[None])  # (1, 512)
    pi_actions = torch.empty(c.horizon, c.num_pi_trajs, c.action_dim)
    zp = z.repeat(c.num_pi_trajs, 1)
    for t in range(c.horizon):  # the policy prior's own sequences
        pi_actions[t], _ = m.pi(zp, noise['pi'][t])
        if t < c.horizon - 1:
            zp = m.dynamics(zp, pi_actions[t])
    z = z.repeat(c.num_samples, 1)
    mean = torch.zeros(c.horizon, c.action_dim)
    if prev_mean is not None:
        mean[:-1] = prev_mean[1:]  # warm start: last step's plan, shifted by one
    std = torch.full((c.horizon, c.action_dim), c.max_std)
    actions = torch.empty(c.horizon, c.num_samples, c.action_dim)
    actions[:, :c.num_pi_trajs] = pi_actions
    for i in range(c.iterations):
        sampled = mean[:, None] + std[:, None] * noise['gauss'][i]  # (3, 488, 6)
        actions[:, c.num_pi_trajs:] = sampled.clamp(-1, 1)
        value = estimate_value(m, z, actions, c, noise['eps'][i], noise['pick'][i]).nan_to_num(0)
        elite = torch.topk(value[:, 0], c.num_elites).indices  # the 64 best
        elite_value, elite_actions = value[elite], actions[:, elite]
        score = torch.exp(c.temperature * (elite_value - elite_value.max(0).values))  # (64, 1)
        score = score / score.sum(0)
        mean = (score[None] * elite_actions).sum(1) / (score.sum(0) + 1e-9)
        std = ((score[None] * (elite_actions - mean[:, None]) ** 2).sum(1) / (score.sum(0) + 1e-9)).sqrt()
        std = std.clamp(c.min_std, c.max_std)
    # Evaluation: the first action of one elite, drawn by Gumbel-max over the weights.
    choice = (score[:, 0].log() + noise['gumbel']).argmax()
    return elite_actions[0, choice], mean


# %% [model] TD-MPC2: encode, plan, act
class TDMPC2(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.encoder = Encoder(c)
        self.dynamics = Dynamics(c)
        self.reward = RewardHead(c)
        self.q = QEnsemble(c)
        self.policy = PolicyPrior(c)

    def pi(self, z, eps):
        return self.policy(z, eps)

    def act(self, obs, noise, prev_mean=None):
        """What the agent does at every step of an episode: plan from the observation, execute the first action,
        and keep the plan's mean to warm-start the next step."""
        return plan(self, obs, self.c, noise, prev_mean)
`;export{n as default};
