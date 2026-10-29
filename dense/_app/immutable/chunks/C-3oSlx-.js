const e=`# Wan 2.1 T2V 1.3B, the text-to-video transformer in the Wan tab, written as plain PyTorch blocks.
# It predicts a flow-matching velocity for a 16 × F × H/8 × W/8 latent of Wan's 3-D causal VAE;
# umT5-XXL encodes the prompt. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 16
    patch: tuple = (1, 2, 2)  # (frames, rows, columns) of latent per token
    d_model: int = 1536
    n_heads: int = 12  # 1536 / 12 = 128 dimensions per head
    n_layers: int = 30
    d_ffn: int = 8960
    text_dim: int = 4096  # umT5-XXL's width
    frequency_dim: int = 256  # cos and sin of t at 128 frequencies
    rope_theta: float = 10000.0
    eps: float = 1e-6


# %% [vae] 3-D causal VAE
class CausalConv3d(nn.Conv3d):
    """The VAE's 3-D convolution. Time is padded on the past side only, so frame f never sees a later
    frame: the first frame is encoded on its own, like a still image."""

    def __init__(self, c_in, c_out, kernel, stride=1, padding=(0, 0, 0)):
        super().__init__(c_in, c_out, kernel, stride=stride)
        pt, ph, pw = padding
        self.pad = (pw, pw, ph, ph, 2 * pt, 0)  # (left, right, top, bottom, past frames, future frames)

    def forward(self, x):  # x: (batch, channels, frames, height, width)
        return super().forward(F.pad(x, self.pad))


def latent_shape(frames, height, width, channels=16):
    """Frame 0 alone, then every 4 frames fold into one latent frame; 8 × smaller in height and width.
    17 frames of 256 × 384 × 3 become 5 × 32 × 48 × 16 numbers."""
    return (channels, 1 + (frames - 1) // 4, height // 8, width // 8)


# %% [patchify] Spatio-temporal patchify
class PatchEmbedding(nn.Module):
    """Cut the latent into 1 × 2 × 2 patches (one latent frame, 2 × 2 positions, 16 channels: 64 numbers)
    and project each to d_model. Tokens run frame by frame, row by row."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Conv3d(c.latent_channels, c.d_model, kernel_size=c.patch, stride=c.patch)

    def forward(self, latent):  # latent: (batch, 16, F, H, W)
        x = self.proj(latent)  # (batch, d_model, F, H/2, W/2)
        grid = tuple(x.shape[2:])
        return x.flatten(2).transpose(1, 2), grid  # (batch, F · H/2 · W/2, d_model)


# %% [rope] 3-D rotary position embedding
def rope_angles(c: Config, grid):
    """Angle of every (token, pair). A head's 128 dimensions form 64 pairs: the first 22 turn with the
    token's frame, the next 21 with its row, the last 21 with its column. Nothing is added to x."""
    head_dim = c.d_model // c.n_heads
    h_dim = w_dim = 2 * (head_dim // 6)  # 42, 42
    t_dim = head_dim - h_dim - w_dim  # 44
    frames, rows, cols = grid
    parts = []
    for dim, count, axis in ((t_dim, frames, 0), (h_dim, rows, 1), (w_dim, cols, 2)):
        frequencies = 1.0 / c.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        angle = torch.arange(count, dtype=torch.float64)[:, None] * frequencies  # (count, dim/2)
        shape = [1, 1, 1, dim // 2]
        shape[axis] = count
        parts.append(angle.view(shape).expand(frames, rows, cols, dim // 2))
    return torch.cat(parts, dim=-1).reshape(frames * rows * cols, head_dim // 2)  # (tokens, 64)


def rotate(x, angles):
    """Turn each pair (x[2i], x[2i+1]) of every head by its angle. x: (batch, tokens, heads, head_dim)."""
    cos = angles.cos()[None, :, None].to(x.dtype)
    sin = angles.sin()[None, :, None].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# %% [conditioning] Timestep and text conditioning
class Conditioning(nn.Module):
    """t → cos/sin → MLP → temb; temb → 6 × d_model of shift, scale and gate shared by all 30 blocks.
    The prompt's umT5 vectors go through their own two-layer MLP to d_model for cross-attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.time_in = nn.Linear(c.frequency_dim, c.d_model)
        self.time_out = nn.Linear(c.d_model, c.d_model)
        self.modulation = nn.Linear(c.d_model, 6 * c.d_model)
        self.text_in = nn.Linear(c.text_dim, c.d_model)
        self.text_out = nn.Linear(c.d_model, c.d_model)

    def sinusoid(self, t):  # t: (batch,) timesteps in [0, 1000)
        half = self.frequency_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        angles = t[:, None].float() * frequencies
        return torch.cat([angles.cos(), angles.sin()], dim=1)  # cos first

    def forward(self, t, text):  # text: (batch, 512, 4096); past the prompt's tokens it is zeros
        temb = self.time_out(F.silu(self.time_in(self.sinusoid(t))))  # (batch, d_model)
        modulation = self.modulation(F.silu(temb)).unflatten(1, (6, -1))  # (batch, 6, d_model)
        text = self.text_out(F.gelu(self.text_in(text), approximate="tanh"))  # (batch, 512, d_model)
        return temb, modulation, text


# %% [attention] Full 3-D self-attention
class SelfAttention(nn.Module):
    """One softmax over every token of every frame: a patch attends to its neighbours, to the same place
    in other frames, and to everything else. q and k are RMS-normalised across all heads, then rotated."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.norm_q = nn.RMSNorm(c.d_model, eps=c.eps)
        self.norm_k = nn.RMSNorm(c.d_model, eps=c.eps)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x, angles):  # x: (batch, tokens, d_model)
        b, n, d = x.shape
        q = rotate(self.norm_q(self.q(x)).view(b, n, self.n_heads, -1), angles).transpose(1, 2)
        k = rotate(self.norm_k(self.k(x)).view(b, n, self.n_heads, -1), angles).transpose(1, 2)
        v = self.v(x).view(b, n, self.n_heads, -1).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (b, heads, n, n)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d))


# %% [cross-attention] Cross-attention to umT5
class CrossAttention(nn.Module):
    """Video tokens ask (queries); the prompt's 512 text vectors answer (keys, values). No mask: the zero
    padding past the prompt becomes 512 − L identical keys. No RoPE on either side."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)  # learned γ, β; no adaLN here
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.norm_q = nn.RMSNorm(c.d_model, eps=c.eps)
        self.norm_k = nn.RMSNorm(c.d_model, eps=c.eps)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x, text):  # x: (batch, tokens, d_model); text: (batch, 512, d_model)
        b, n, d = x.shape
        q = self.norm_q(self.q(self.norm(x))).view(b, n, self.n_heads, -1).transpose(1, 2)
        k = self.norm_k(self.k(text)).view(b, -1, self.n_heads, q.shape[-1]).transpose(1, 2)
        v = self.v(text).view(b, -1, self.n_heads, q.shape[-1]).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (b, heads, n, 512)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d))


# %% [mlp] Feed-forward
class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_ffn)
        self.down = nn.Linear(c.d_ffn, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] Wan block
class Block(nn.Module):
    """adaLN self-attention, plain cross-attention, adaLN feed-forward. The six vectors are the shared
    modulation plus this block's own learned table."""

    def __init__(self, c: Config):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(1, 6, c.d_model))
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps, elementwise_affine=False)
        self.attention = SelfAttention(c)
        self.cross = CrossAttention(c)
        self.norm3 = nn.LayerNorm(c.d_model, eps=c.eps, elementwise_affine=False)
        self.mlp = FeedForward(c)

    def forward(self, x, text, modulation, angles):
        shift1, scale1, gate1, shift2, scale2, gate2 = (self.table + modulation).chunk(6, dim=1)
        x = x + gate1 * self.attention(self.norm1(x) * (1 + scale1) + shift1, angles)
        x = x + self.cross(x, text)
        return x + gate2 * self.mlp(self.norm3(x) * (1 + scale2) + shift2)


# %% [unpatchify] Output and unpatchify
class Output(nn.Module):
    """adaLN from temb (shift and scale, no gate), a Linear to 1 × 2 × 2 × 16 numbers per token, then the
    tokens are put back into a 16 × F × H × W velocity."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.table = nn.Parameter(torch.zeros(1, 2, c.d_model))
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps, elementwise_affine=False)
        self.proj = nn.Linear(c.d_model, c.latent_channels * math.prod(c.patch))

    def forward(self, x, temb, grid):  # x: (batch, tokens, d_model)
        shift, scale = (self.table + temb[:, None]).chunk(2, dim=1)
        x = self.proj(self.norm(x) * (1 + scale) + shift)
        (pt, ph, pw), (f, h, w) = self.c.patch, grid
        x = x.reshape(-1, f, h, w, pt, ph, pw, self.c.latent_channels)
        return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(-1, self.c.latent_channels, f * pt, h * ph, w * pw)


# %% [flow-matching] Flow matching
def sigmas(steps, shift=3.0, train_steps=1000):
    """Noise levels σ from 1 (pure noise) down to 0. Flow matching mixes x_σ = (1 − σ) x₀ + σ ε, so the
    model predicts the straight-line velocity v = ε − x₀. Shift 3 spends more steps at high noise.
    diffusers shifts the endpoints once at construction and again in set_timesteps; this follows it."""
    shifted = lambda s: shift * s / (1 + (shift - 1) * s)  # noqa: E731
    lowest = shifted(torch.tensor(1 / train_steps, dtype=torch.float32)).item()  # the training grid's last σ
    s = torch.linspace(train_steps, lowest * train_steps, steps, dtype=torch.float64) / train_steps
    return torch.cat([shifted(s).float(), torch.zeros(1)])  # σ₀ … σ_steps = 0; the model sees t = 1000 σ


def guided_velocity(v_text, v_negative, scale=5.0):
    """Classifier-free guidance: start from the negative prompt's velocity, move w times towards the prompt's."""
    return v_negative + scale * (v_text - v_negative)


def euler_step(x, v, sigma, sigma_next):
    """One Euler step along the line: x_σ' = x_σ + (σ' − σ) v. With σ' = 0 it lands on x₀ = x_σ − σ v."""
    return x + (sigma_next - sigma) * v


# %% [model] Wan transformer
class Wan(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.conditioning = Conditioning(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)

    def forward(self, latent, t, text):  # latent: (batch, 16, F, H, W); t: (batch,); text: (batch, 512, 4096)
        x, grid = self.patchify(latent)
        angles = rope_angles(self.c, grid)
        temb, modulation, text = self.conditioning(t, text)
        for block in self.blocks:
            x = block(x, text, modulation, angles)
        return self.output(x, temb, grid)  # the velocity v, same shape as the latent
`;export{e as default};
