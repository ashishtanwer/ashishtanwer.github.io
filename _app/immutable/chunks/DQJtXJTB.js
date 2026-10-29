const n=`# DIAMOND (Alonso et al. 2024), eloialonso/diamond atari_100k/models/Breakout.pt, the world model as plain PyTorch.
# A small pixel U-Net denoises the next 64 × 64 frame of the game, reading the last four frames (stacked as
# channels) and the last four actions. EDM's preconditioning lets three Euler steps sample a frame; repeating that
# with the frame appended to the history plays the game in imagination.
# Each "# %% [id]" section is one step of the DIAMOND tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    image_size: int = 64
    channels: int = 3  # RGB in [-1, 1]
    context: int = 4  # past frames and actions the denoiser reads
    actions: int = 4  # Breakout: NOOP, FIRE, RIGHT, LEFT
    cond: int = 256  # width of the conditioning vector
    widths: tuple = (64, 64, 64, 64)  # channels at 64, 32, 16 and 8 px
    depths: tuple = (2, 2, 2, 2)  # ResNet blocks per level going down; one more going up
    head_dim: int = 8  # attention (in the middle only): 64 channels in 8 heads
    group_size: int = 32  # GroupNorm: one group per 32 channels
    eps: float = 1e-5
    sigma_data: float = 0.5  # the standard deviation of the frames
    sigma_offset_noise: float = 0.3  # extra per-channel noise seen in training
    steps: int = 3  # denoising steps per frame
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: float = 7.0


# %% [conditioning] Noise level and actions
class Conditioning(nn.Module):
    """c_noise goes through fixed random Fourier features; each of the four past actions has a learned 64-wide
    vector, and the four are laid side by side (256). Their sum, through an MLP, is the vector every block reads."""

    def __init__(self, c: Config):
        super().__init__()
        self.register_buffer('frequencies', torch.randn(1, c.cond // 2))  # drawn once, never trained
        self.actions = nn.Embedding(c.actions, c.cond // c.context)
        self.mlp = nn.Sequential(nn.Linear(c.cond, c.cond), nn.SiLU(), nn.Linear(c.cond, c.cond))

    def forward(self, c_noise, act):  # c_noise: (batch,), act: (batch, 4) action ids, oldest first
        angles = 2 * math.pi * c_noise[:, None] @ self.frequencies  # (batch, 128)
        noise = torch.cat([angles.cos(), angles.sin()], dim=-1)  # (batch, 256)
        actions = self.actions(act).flatten(1)  # (batch, 4 × 64)
        return self.mlp(noise + actions)  # (batch, 256)


# %% [resblock] ResNet block with adaptive GroupNorm
class AdaGroupNorm(nn.Module):
    """GroupNorm without its own scale and shift; they come from the conditioning vector instead, one pair per channel."""

    def __init__(self, width: int, c: Config):
        super().__init__()
        self.groups, self.eps = max(1, width // c.group_size), c.eps
        self.linear = nn.Linear(c.cond, 2 * width)

    def forward(self, x, cond):
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)  # (batch, width, 1, 1) each
        return F.group_norm(x, self.groups, eps=self.eps) * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, c: Config, attention: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()  # skip path
        self.norm1 = AdaGroupNorm(c_in, c)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = AdaGroupNorm(c_out, c)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.attn = SelfAttention(c_out, c) if attention else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return self.attn(self.proj(x) + h)


# %% [attention] Self-attention at 8 × 8
class SelfAttention(nn.Module):
    """In the two middle blocks, each of the 64 pixels attends to all 64 in 8 heads of 8 channels. As in the
    reference code, the residual is added to the normalized input."""

    def __init__(self, width: int, c: Config):
        super().__init__()
        self.heads = max(1, width // c.head_dim)
        self.norm = nn.GroupNorm(max(1, width // c.group_size), width, eps=c.eps)
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.out = nn.Conv2d(width, width, 1)

    def forward(self, x):
        b, width, h, w = x.shape
        x = self.norm(x)
        q, k, v = self.qkv(x).view(b, 3 * self.heads, width // self.heads, h * w).transpose(2, 3).chunk(3, dim=1)
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)  # (batch, 8, 64, 64)
        y = (weights @ v).transpose(2, 3).reshape(b, width, h, w)
        return x + self.out(y)


# %% [unet] U-Net over stacked frames
class Denoiser(nn.Module):
    """F_θ: the four past frames and the noisy next frame enter side by side as 15 channels. Down 64 → 32 → 16 → 8,
    two blocks in the middle with attention, then up again; each up block concatenates a saved down output."""

    def __init__(self, c: Config):
        super().__init__()
        w, d = c.widths, c.depths
        self.conditioning = Conditioning(c)
        self.conv_in = nn.Conv2d((c.context + 1) * c.channels, w[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i, n in enumerate(d):
            c1, c2 = w[max(0, i - 1)], w[i]
            self.down.append(nn.ModuleList([ResBlock(c1 if j == 0 else c2, c2, c) for j in range(n)]))
            ins, outs = [2 * c2] * n + [c1 + c2], [c2] * n + [c1]  # each input: x next to a saved output
            self.up.insert(0, nn.ModuleList([ResBlock(a, b, c) for a, b in zip(ins, outs)]))
        self.mid = nn.ModuleList([ResBlock(w[-1], w[-1], c, attention=True) for _ in range(2)])
        self.downsamples = nn.ModuleList([nn.Identity()] + [nn.Conv2d(ch, ch, 3, stride=2, padding=1) for ch in w[:-1]])
        self.upsamples = nn.ModuleList([nn.Identity()] + [nn.Conv2d(ch, ch, 3, padding=1) for ch in reversed(w[:-1])])
        self.norm_out = nn.GroupNorm(max(1, w[0] // c.group_size), w[0], eps=c.eps)
        self.conv_out = nn.Conv2d(w[0], c.channels, 3, padding=1)

    def forward(self, noisy, c_noise, frames, act):  # noisy: (b, 3, 64, 64), frames: (b, 12, 64, 64)
        cond = self.conditioning(c_noise, act)
        x = self.conv_in(torch.cat([frames, noisy], dim=1))
        saved = []
        for blocks, down in zip(self.down, self.downsamples):
            x = down(x)
            level = [x]
            for block in blocks:
                x = block(x, cond)
                level.append(x)
            saved.append(level)
        for block in self.mid:
            x = block(x, cond)
        for blocks, conv, level in zip(self.up, self.upsamples, reversed(saved)):
            if not isinstance(conv, nn.Identity):
                x = conv(F.interpolate(x, scale_factor=2.0, mode='nearest'))
            for block, skip in zip(blocks, reversed(level)):  # the newest saved output first
                x = block(torch.cat([x, skip], dim=1), cond)
        return self.conv_out(F.silu(self.norm_out(x)))  # (b, 3, 64, 64)


# %% [preconditioning] EDM preconditioning
def conditioners(sigma, c: Config):
    """EDM's scalings, at the noise level including the offset noise: c_in keeps the input at unit variance,
    c_skip and c_out mix the noisy input with the network's output, c_noise tells the network σ."""
    sigma = (sigma**2 + c.sigma_offset_noise**2).sqrt()
    c_in = 1 / (sigma**2 + c.sigma_data**2).sqrt()
    c_skip = c.sigma_data**2 / (sigma**2 + c.sigma_data**2)
    c_out = sigma * c_skip.sqrt()
    c_noise = sigma.log() / 4
    return c_in, c_out, c_skip, c_noise


@torch.no_grad()
def denoise(net, noisy, sigma, frames, act, c: Config):
    """D(x; σ) = c_skip · x + c_out · F_θ(c_in · x, c_noise, frames / σ_data, actions), rounded to 8-bit pixels."""
    c_in, c_out, c_skip, c_noise = conditioners(torch.full((noisy.shape[0],), float(sigma)), c)
    out = net(noisy * c_in[:, None, None, None], c_noise, frames / c.sigma_data, act)
    d = c_skip[:, None, None, None] * noisy + c_out[:, None, None, None] * out
    return (d.clamp(-1, 1) + 1).div(2).mul(255).floor().div(255).mul(2).sub(1)  # 256 levels per channel, like a frame


# %% [sampler] Three Euler steps
def sigmas(c: Config):
    """Karras spacing from σ_max = 5 to σ_min = 0.002 in 3 levels, then 0."""
    lo, hi = c.sigma_min ** (1 / c.rho), c.sigma_max ** (1 / c.rho)
    s = (hi + torch.linspace(0, 1, c.steps) * (lo - hi)) ** c.rho
    return torch.cat([s, s.new_zeros(1)])


@torch.no_grad()
def sample_frame(net, frames, act, noise, c: Config):
    """Start from unit Gaussian noise (the reference code does not scale it by σ_max) and take Euler steps along
    dx/dσ = (x − D(x; σ)) / σ. The last step goes to σ = 0, so the frame is D's 8-bit output."""
    x, trajectory = noise, [noise]
    s = sigmas(c)
    for sigma, sigma_next in zip(s[:-1], s[1:]):
        d = (x - denoise(net, x, sigma, frames, act, c)) / sigma
        x = x + d * (sigma_next - sigma)
        trajectory.append(x)
    return x, trajectory


# %% [model] Playing in imagination
@torch.no_grad()
def rollout(net, frames, actions, new_actions, noises, c: Config):
    """frames: (b, 4, 3, 64, 64) and actions: (b, 4), the last four of each. For every new action: it replaces the
    last action (the one taken at the newest frame), a frame is sampled, and both histories shift by one."""
    out = []
    for action, noise in zip(new_actions, noises):
        actions = torch.cat([actions[:, :-1], action[:, None]], dim=1)
        frame, _ = sample_frame(net, frames.flatten(1, 2), actions, noise, c)
        frames = torch.cat([frames[:, 1:], frame[:, None]], dim=1)
        actions = actions.roll(-1, dims=1)  # the slot freed at the end is filled by the next action
        out.append(frame)
    return torch.stack(out, dim=1)  # (b, steps, 3, 64, 64)
`;export{n as default};
