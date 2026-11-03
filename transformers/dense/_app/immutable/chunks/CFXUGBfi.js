const n=`# PPO on LunarLander-v3, as the RL Explainer trains it on the CPU: an actor and a critic (two 64 × 64 tanh MLPs),
# generalised advantage estimates along each trajectory, and the clipped surrogate objective. scripts/rl/train_ppo.py
# trains with these blocks; the check recomputes the recorded advantages and one recorded minibatch step.
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class Config:
    obs_dim: int = 8  # x, y, vx, vy, angle, angular velocity, left leg down, right leg down
    actions: int = 4  # nothing, left engine, main engine, right engine
    hidden: int = 64
    gamma: float = 0.999
    lam: float = 0.98  # GAE's λ
    clip: float = 0.2  # ε: the ratio is clipped to [1 − ε, 1 + ε]
    epochs: int = 4
    minibatch: int = 64
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def mlp(c: Config, out: int, gain: float):
    layers = [nn.Linear(c.obs_dim, c.hidden), nn.Tanh(), nn.Linear(c.hidden, c.hidden), nn.Tanh(), nn.Linear(c.hidden, out)]
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, 2**0.5 if layer is not layers[-1] else gain)
            nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers)


# %% [policy] The actor and the critic
class ActorCritic(nn.Module):
    """π(a | s) as four logits, and V(s), the return the critic expects from s under the current policy."""

    def __init__(self, c: Config):
        super().__init__()
        self.actor = mlp(c, c.actions, 0.01)  # a near-uniform policy at the start
        self.critic = mlp(c, 1, 1.0)

    def dist(self, s):
        return Categorical(logits=self.actor(s))

    def value(self, s):  # (batch, 8) → (batch,)
        return self.critic(s).squeeze(-1)


# %% [gae] Generalised advantage estimates
def gae(rewards, values, next_values, last, gamma, lam):
    """Along each environment's steps (time first): δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ), Aₜ = δₜ + γλ·Aₜ₊₁, the sum cut where an
    episode ended (last). next_values holds V(sₜ₊₁): 0 after a crash or a landing, the critic's value of the final
    state after a time-out, and of the next state otherwise (also past the end of the rollout)."""
    deltas = rewards + gamma * next_values - values
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        running = deltas[t] + gamma * lam * (1 - last[t]) * running
        advantages[t] = running
    return advantages, deltas


# %% [ppo-loss] The clipped surrogate
def ppo_loss(model, s, a, old_logp, advantages, returns, c: Config):
    """L = −mean(min(ρ·Â, clip(ρ, 1 − ε, 1 + ε)·Â)) + c_v·(V − R)² − c_e·entropy, with ρ = π(a|s) / π_old(a|s) and the
    advantages normalised over the minibatch. Where the clip is the smaller side, ρ gets no gradient."""
    dist = model.dist(s)
    logp = dist.log_prob(a)
    ratio = (logp - old_logp).exp()
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surrogate = torch.min(ratio * adv, ratio.clamp(1 - c.clip, 1 + c.clip) * adv)
    policy_loss = -surrogate.mean()
    value_loss = ((model.value(s) - returns) ** 2).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + c.vf_coef * value_loss - c.ent_coef * entropy
    with torch.no_grad():
        stats = {'ratio': ratio, 'adv': adv, 'policyLoss': policy_loss, 'valueLoss': value_loss, 'entropy': entropy,
                 'clipped': ((ratio - 1).abs() > c.clip).float(), 'approxKl': ((ratio - 1) - (logp - old_logp)).mean()}
    return loss, stats


# %% [ppo-update] One update: epochs over shuffled minibatches
def ppo_update(model, optimizer, rollout, c: Config, generator, on_minibatch=None):
    """Every sample of the rollout, c.epochs times, in minibatches of c.minibatch; one clipped Adam step each."""
    n = len(rollout['s'])
    for epoch in range(c.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, c.minibatch):
            idx = order[start:start + c.minibatch]
            mb = {k: v[idx] for k, v in rollout.items()}
            loss, stats = ppo_loss(model, mb['s'], mb['a'], mb['logp'], mb['adv'], mb['ret'], c)
            if on_minibatch:
                on_minibatch(epoch, start // c.minibatch, idx, mb, stats)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
            optimizer.step()
`;export{n as default};
