const e=`# DreamerV3 (Hafner et al. 2023/2025), danijar/dreamerv3, the world model and actor-critic as plain PyTorch.
# A CNN encodes each 64 × 64 frame. The RSSM keeps a deterministic state h (a block GRU) and a stochastic state z:
# 32 categorical variables of 32 classes. From h alone it predicts a prior over z; with the frame's encoding it forms
# the posterior. A decoder rebuilds the frame from (h, z), and heads predict the reward and the value as two-hot
# distributions over symlog-spaced bins, and whether the episode continues. The actor and critic learn only from
# trajectories the model imagines, 15 steps from every replayed state.
# Each "# %% [id]" section is one step of the DreamerV3 tab's diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    image: int = 64
    actions: int = 1  # cartpole: one continuous force in [-1, 1]
    deter: int = 512  # h
    hidden: int = 64
    stoch: int = 32  # z: 32 variables …
    classes: int = 32  # … of 32 classes each
    blocks: int = 8  # the GRU is block-diagonal: 8 independent blocks
    depth: int = 4  # CNN channels: depth × (2, 3, 4, 4)
    mults: tuple = (2, 3, 4, 4)
    units: int = 64  # MLP width of the heads
    kernel: int = 5
    bspace: int = 8  # the decoder's block-linear groups
    unimix: float = 0.01  # 1% uniform mixed into every categorical
    bins: int = 255
    minstd: float = 0.1
    maxstd: float = 1.0
    eps: float = 1e-4  # RMSNorm
    imag_length: int = 15
    lam: float = 0.95
    actent: float = 3e-4


class RMSNorm(nn.Module):
    """x · rsqrt(mean(x²) + ε) · scale over the last axis (DreamerV3 normalizes everything this way)."""

    def __init__(self, width: int, eps: float):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class BlockLinear(nn.Module):
    """A linear layer split into \`blocks\` independent pieces: block k maps its slice of the input to its slice of
    the output, so h's 8 blocks only mix through the shared inputs."""

    def __init__(self, d_in: int, d_out: int, blocks: int):
        super().__init__()
        self.blocks = blocks
        self.kernel = nn.Parameter(torch.zeros(blocks, d_in // blocks, d_out // blocks))
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        x = x.view(*x.shape[:-1], self.blocks, -1)  # (..., blocks, d_in / blocks)
        return torch.einsum('...ki,kio->...ko', x, self.kernel).flatten(-2) + self.bias


def layer(d_in: int, d_out: int, c: Config) -> nn.Sequential:
    """Linear → RMSNorm → SiLU, the unit of every DreamerV3 MLP."""
    return nn.Sequential(nn.Linear(d_in, d_out), RMSNorm(d_out, c.eps), nn.SiLU())


# %% [symlog] symlog, symexp and two-hot over 255 bins
def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * torch.expm1(x.abs())


def twohot_bins(c: Config):
    """255 bins at symexp(linspace(−20, 20)): dense near 0, sparse far out, symmetric around 0."""
    half = symexp(torch.linspace(-20, 0, (c.bins - 1) // 2 + 1))
    return torch.cat([half, -half[:-1].flip(0)])


def twohot_read(logits, bins):
    """The prediction: the probability-weighted mean of the bins' real values (summed symmetrically, as the
    reference does, so uniform probabilities read exactly 0)."""
    p = F.softmax(logits, dim=-1)
    m = (bins.shape[0] - 1) // 2
    left, mid, right = p[..., :m] * bins[:m], p[..., m] * bins[m], p[..., m + 1:] * bins[m + 1:]
    return mid + (left.flip(-1) + right).sum(-1)


def twohot_loss(logits, target, bins):
    """Cross-entropy with the target's two-hot encoding: its two neighbouring bins, weighted by distance."""
    below = ((bins <= target[..., None]).sum(-1) - 1).clamp(0, len(bins) - 1)
    above = (len(bins) - (bins > target[..., None]).sum(-1)).clamp(0, len(bins) - 1)
    equal = below == above
    d_below = torch.where(equal, torch.ones_like(target), (bins[below] - target).abs())
    d_above = torch.where(equal, torch.ones_like(target), (bins[above] - target).abs())
    total = d_below + d_above
    two_hot = (F.one_hot(below, len(bins)) * (d_above / total)[..., None]
               + F.one_hot(above, len(bins)) * (d_below / total)[..., None])
    return -(two_hot * F.log_softmax(logits, dim=-1)).sum(-1)


# %% [encoder] CNN encoder
class Encoder(nn.Module):
    """64 × 64 × 3 → four stages of (5 × 5 conv, 2 × 2 max-pool, RMSNorm, SiLU) → a 4 × 4 map, flattened."""

    def __init__(self, c: Config):
        super().__init__()
        widths = [3] + [c.depth * m for m in c.mults]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, c.kernel, padding=c.kernel // 2) for a, b in zip(widths, widths[1:]))
        self.norms = nn.ModuleList(RMSNorm(b, c.eps) for b in widths[1:])

    def forward(self, image):  # (batch, 64, 64, 3) uint8 → (batch, 4·4·channels)
        x = (image.float() / 255 - 0.5).permute(0, 3, 1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = F.max_pool2d(conv(x), 2)
            x = F.silu(norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))  # normalize over channels
        return x.permute(0, 2, 3, 1).flatten(1)  # height, width, channel order, as the reference


# %% [rssm] RSSM core: the block GRU that carries h
class Core(nn.Module):
    """(h, z, a) → the next h. h, z and a are each embedded (Linear–RMSNorm–SiLU); each of the 8 blocks of the GRU
    reads its own slice of h plus all three embeddings, so the blocks only mix through those."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.dynin0 = layer(c.deter, c.hidden, c)
        self.dynin1 = layer(c.stoch * c.classes, c.hidden, c)
        self.dynin2 = layer(c.actions, c.hidden, c)
        self.dynhid0 = BlockLinear(c.deter + 3 * c.hidden * c.blocks, c.deter, c.blocks)
        self.dynhid0norm = RMSNorm(c.deter, c.eps)
        self.dyngru = BlockLinear(c.deter, 3 * c.deter, c.blocks)

    def forward(self, h, z, action):
        c, g = self.c, self.c.blocks
        action = action / action.abs().clamp(min=1)
        x = torch.cat([self.dynin0(h), self.dynin1(z.flatten(-2)), self.dynin2(action)], -1)  # (batch, 3·hidden)
        x = torch.cat([h.view(-1, g, c.deter // g), x[:, None].expand(-1, g, -1)], -1).flatten(1)
        x = F.silu(self.dynhid0norm(self.dynhid0(x)))
        reset, cand, update = self.dyngru(x).view(-1, g, 3, c.deter // g).unbind(2)  # per block: 3 gates
        reset, cand, update = reset.flatten(1), cand.flatten(1), update.flatten(1)
        cand = torch.tanh(torch.sigmoid(reset) * cand)
        update = torch.sigmoid(update - 1)  # biased toward keeping the old state
        return update * cand + (1 - update) * h


# %% [posterior] Posterior and prior over the 32 × 32 categorical z
class Posterior(nn.Module):
    """With the frame: logits of z from h and the encoder's tokens."""

    def __init__(self, c: Config, token_dim: int):
        super().__init__()
        self.c = c
        self.obs0 = layer(c.deter + token_dim, c.hidden, c)
        self.obslogit = nn.Linear(c.hidden, c.stoch * c.classes)

    def forward(self, h, tokens):
        return self.obslogit(self.obs0(torch.cat([h, tokens], -1))).view(-1, self.c.stoch, self.c.classes)


class Prior(nn.Module):
    """Without the frame: what h alone predicts about z. Imagination samples from this."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.prior0, self.prior1 = layer(c.deter, c.hidden, c), layer(c.hidden, c.hidden, c)
        self.priorlogit = nn.Linear(c.hidden, c.stoch * c.classes)

    def forward(self, h):
        return self.priorlogit(self.prior1(self.prior0(h))).view(-1, self.c.stoch, self.c.classes)


def probs(logits, c: Config):
    """1% uniform mixed in, so no class ever has probability 0 (and the KL stays finite)."""
    return (1 - c.unimix) * F.softmax(logits, -1) + c.unimix / c.classes


def sample(logits, gumbel, c: Config):
    """One class per variable (Gumbel-max with the given noise), as one-hot vectors."""
    return F.one_hot((probs(logits, c).log() + gumbel).argmax(-1), c.classes).float()


def kl(post_logits, prior_logits, c: Config):
    """KL(posterior ‖ prior), summed over the 32 variables; the dyn and rep losses clip it below at 1 nat."""
    p, q = probs(post_logits, c), probs(prior_logits, c)
    return (p * (p.log() - q.log())).sum((-1, -2))


# %% [decoder] Decoder: (h, z) → the frame
class Decoder(nn.Module):
    """h through a block-linear map and z through an MLP make a 4 × 4 map; four (upsample ×2, 5 × 5 conv) stages
    bring it back to 64 × 64 × 3, and a sigmoid gives pixel values in [0, 1]."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.widths = [c.depth * m for m in c.mults]
        top, cells = self.widths[-1], 4 * 4
        self.sp0 = BlockLinear(c.deter, cells * top, c.bspace)
        self.sp1 = layer(c.stoch * c.classes, 2 * c.units, c)
        self.sp2 = nn.Linear(2 * c.units, cells * top)
        self.spnorm = RMSNorm(top, c.eps)
        ins = self.widths[::-1]
        outs = self.widths[:-1][::-1]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, c.kernel, padding=c.kernel // 2) for a, b in zip(ins, outs))
        self.norms = nn.ModuleList(RMSNorm(b, c.eps) for b in outs)
        self.imgout = nn.Conv2d(self.widths[0], 3, c.kernel, padding=c.kernel // 2)

    def forward(self, h, z):  # → (batch, 64, 64, 3) in [0, 1]
        g, top = self.c.bspace, self.widths[-1]
        x0 = self.sp0(h).view(-1, g, 4, 4, top // g).permute(0, 2, 3, 1, 4).flatten(3)  # (g h w c) → h w (g c)
        x1 = self.sp2(self.sp1(z.flatten(-2))).view(-1, 4, 4, top)
        x = F.silu(self.spnorm(x0 + x1)).permute(0, 3, 1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(F.interpolate(x, scale_factor=2, mode='nearest'))
            x = F.silu(norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = self.imgout(F.interpolate(x, scale_factor=2, mode='nearest'))
        return torch.sigmoid(x).permute(0, 2, 3, 1)


# %% [heads] Reward, continue, value and policy heads
class Heads(nn.Module):
    """Each reads the model state (h, z flattened): one Linear–RMSNorm–SiLU layer for the reward and continue
    heads, three for the policy and the value."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        feat = c.deter + c.stoch * c.classes
        self.rew = nn.Sequential(layer(feat, c.units, c), nn.Linear(c.units, c.bins))
        self.con = nn.Sequential(layer(feat, c.units, c), nn.Linear(c.units, 1))
        self.val = nn.Sequential(layer(feat, c.units, c), layer(c.units, c.units, c), layer(c.units, c.units, c),
                                 nn.Linear(c.units, c.bins))
        self.pol = nn.Sequential(layer(feat, c.units, c), layer(c.units, c.units, c), layer(c.units, c.units, c))
        self.pol_mean, self.pol_std = nn.Linear(c.units, c.actions), nn.Linear(c.units, c.actions)

    def policy(self, feat):
        """A Gaussian with mean tanh(·) and a standard deviation squeezed into [0.1, 1]."""
        x = self.pol(feat)
        std = (self.c.maxstd - self.c.minstd) * torch.sigmoid(self.pol_std(x) + 2) + self.c.minstd
        return torch.tanh(self.pol_mean(x)), std

    def forward(self, feat, bins):
        return dict(reward=twohot_read(self.rew(feat), bins), cont=torch.sigmoid(self.con(feat)[..., 0]),
                    value=twohot_read(self.val(feat), bins))


# %% [imagine] Imagination: rolling the RSSM forward without frames
@torch.no_grad()
def imagine(core: Core, prior: Prior, h, z, actions, gumbel, c: Config):
    """From a model state, feed actions (one per step) through the core and sample z from the prior each time.
    \`actions\` may be a function of the state (the policy) or a (steps, batch, actions) tensor."""
    hs, zs, logits, taken = [], [], [], []
    for k in range(len(gumbel)):
        a = actions(h, z, k) if callable(actions) else actions[k]
        h = core(h, z, a)
        logit = prior(h)
        z = sample(logit, gumbel[k], c)
        hs.append(h), zs.append(z), logits.append(logit), taken.append(a)
    return torch.stack(hs), torch.stack(zs), torch.stack(logits), torch.stack(taken)


# %% [actor-critic] λ-returns and the actor-critic losses
def lambda_return(reward, cont, value, lam):
    """R_t = r_{t+1} + c_{t+1}((1 − λ) v_{t+1} + λ R_{t+1}), from the end of the imagined trajectory backwards.
    Inputs are (batch, steps + 1): the start state and the 15 imagined ones."""
    rets = [value[:, -1]]
    interm = reward[:, 1:] + (1 - lam) * cont[:, 1:] * value[:, 1:]
    for t in reversed(range(reward.shape[1] - 1)):
        rets.append(interm[:, t] + cont[:, t + 1] * lam * rets[-1])
    return torch.stack(rets[::-1][:-1], 1)


def actor_critic_losses(action, mean, std, reward, cont, value_logits, value, slow_value, ret_scale, bins,
                        c: Config):
    """The losses imag_loss computes on an imagined trajectory (batch, 16), given the returns' scale:
    actor: REINFORCE on the advantage (λ-return − value) / scale, plus a small entropy bonus;
    critic: cross-entropy of its two-hot distribution with the λ-return, and with the slow critic's prediction.
    The returns bootstrap from the critic itself (imag_loss.slowtar is off by default)."""
    weight = torch.cumprod(cont, 1)  # later steps count less once the episode may have ended
    ret = lambda_return(reward, cont, value, c.lam)
    adv = (ret - value[:, :-1]) / ret_scale
    dist = torch.distributions.Normal(mean, std)
    logpi = dist.log_prob(action).sum(-1)[:, :-1]
    entropy = dist.entropy().sum(-1)[:, :-1]
    policy = weight[:, :-1] * -(logpi * adv + c.actent * entropy)
    target = torch.cat([ret, torch.zeros_like(ret[:, -1:])], 1)
    critic = weight[:, :-1] * (twohot_loss(value_logits, target, bins)
                               + twohot_loss(value_logits, slow_value, bins))[:, :-1]
    return policy, critic, ret


# %% [model] One step of the world model
class WorldModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.encoder = Encoder(c)
        token_dim = 4 * 4 * c.depth * c.mults[-1]
        self.core = Core(c)
        self.posterior = Posterior(c, token_dim)
        self.prior = Prior(c)
        self.decoder = Decoder(c)
        self.heads = Heads(c)
        self.register_buffer('bins', twohot_bins(c))

    @torch.no_grad()
    def observe(self, h, z, action, image, gumbel):
        """A real step: the core moves h with the last z and action, the frame's tokens give the posterior, z is
        sampled from it; the decoder and heads read the new (h, z)."""
        h = self.core(h, z, action)
        post = self.posterior(h, self.encoder(image))
        z = sample(post, gumbel, self.c)
        feat = torch.cat([h, z.flatten(-2)], -1)
        return h, z, post, self.prior(h), self.decoder(h, z), self.heads(feat, self.bins)
`;export{e as default};
