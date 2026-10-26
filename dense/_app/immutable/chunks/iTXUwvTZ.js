const e=`# DiT-XL/2-256, the class-conditional diffusion transformer in the DiT tab, written as plain PyTorch blocks.
# It predicts the noise in a 4 × 32 × 32 VAE latent; the bundled VAE turns the final latent into 256 × 256 pixels.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 4
    latent_size: int = 32  # 256 px / the VAE's factor of 8
    patch_size: int = 2  # 16 × 16 = 256 patches of 2 × 2 × 4 numbers
    d_model: int = 1152
    n_heads: int = 16  # 1152 / 16 = 72 dimensions per head
    n_layers: int = 28
    d_mlp: int = 4608  # 4 × d_model
    n_classes: int = 1000  # ImageNet; row 1000 of the class table is the null class
    frequency_dim: int = 256  # cos and sin of t at 128 frequencies
    out_channels: int = 8  # predicted noise (4) and a variance that DDIM does not use (4)


# %% [patchify] Patch embedding
def sincos(d, positions):
    """d/2 sines then d/2 cosines of each position at frequencies 10000^(−2i/d)."""
    frequencies = 10000 ** (-torch.arange(d // 2, dtype=torch.float64) / (d / 2))
    angles = positions[:, None].double() * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class PatchEmbedding(nn.Module):
    """Cut the latent into 2 × 2 patches, project each to d_model, add a fixed 2D sin-cos position."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Conv2d(c.latent_channels, c.d_model, kernel_size=c.patch_size, stride=c.patch_size)
        side = c.latent_size // c.patch_size
        rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        # First half of each vector encodes the patch's column, second half its row; not learned.
        position = torch.cat([sincos(c.d_model // 2, cols.flatten()), sincos(c.d_model // 2, rows.flatten())], dim=1)
        self.register_buffer("position", position.float(), persistent=False)  # (256, d_model)

    def forward(self, latent):  # latent: (batch, 4, 32, 32)
        patches = self.proj(latent).flatten(2).transpose(1, 2)  # (batch, 256, d_model), row by row
        return patches + self.position


# %% [timestep-embedding] Timestep and class embedding
class Conditioning(nn.Module):
    """c = MLP(sin/cos of t) + class row. One vector per image, shared by every block."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.mlp_in = nn.Linear(c.frequency_dim, c.d_model)
        self.mlp_out = nn.Linear(c.d_model, c.d_model)
        self.classes = nn.Embedding(c.n_classes + 1, c.d_model)  # + the null class, for guidance

    def sinusoid(self, t):  # t: (batch,) integer timesteps 0…999
        half = self.frequency_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        angles = t[:, None].float() * frequencies
        return torch.cat([angles.cos(), angles.sin()], dim=1)  # (batch, 256), cos first

    def forward(self, t, class_ids):
        timestep = self.mlp_out(F.silu(self.mlp_in(self.sinusoid(t))))
        return timestep + self.classes(class_ids)  # c: (batch, d_model)


# %% [adaln-zero] adaLN-Zero
class AdaLNZero(nn.Module):
    """From c, six vectors per block: shift β, scale γ and gate α for attention and for the MLP.
    "Zero": training starts this Linear at zero, so every gate is 0 and every block starts as the identity."""

    def __init__(self, c: Config):
        super().__init__()
        self.linear = nn.Linear(c.d_model, 6 * c.d_model)

    def forward(self, cond):  # cond: (batch, d_model)
        return self.linear(F.silu(cond))[:, None].chunk(6, dim=-1)  # six (batch, 1, d_model)


def modulate(x, norm, shift, scale):
    """LayerNorm without learned γ, β; the conditioning supplies them instead."""
    return norm(x) * (1 + scale) + shift


# %% [attention] Multi-head self-attention
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, 256 patches, d_model)
        b, t, d = x.shape
        q, k, v = (proj(x).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # every patch sees every patch
        heads = weights @ v  # (batch, heads, 256, head_dim)
        return self.out(heads.transpose(1, 2).reshape(b, t, d))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] DiT block
class Block(nn.Module):
    """x + α₁ · Attention(adaLN(x)), then x + α₂ · MLP(adaLN(x)), with β, γ, α from c."""

    def __init__(self, c: Config):
        super().__init__()
        self.ada = AdaLNZero(c)
        self.norm1 = nn.LayerNorm(c.d_model, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(c)
        self.norm2 = nn.LayerNorm(c.d_model, elementwise_affine=False, eps=1e-5)  # diffusers' value; DiT's own is 1e-6
        self.mlp = MLP(c)

    def forward(self, x, cond):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond)
        x = x + gate1 * self.attention(modulate(x, self.norm1, shift1, scale1))
        return x + gate2 * self.mlp(modulate(x, self.norm2, shift2, scale2))


# %% [unpatchify] Final layer and unpatchify
class FinalLayer(nn.Module):
    """adaLN (shift and scale, no gate), a Linear to 2 × 2 × 8 numbers per patch, then patches back to a grid."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.norm = nn.LayerNorm(c.d_model, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Linear(c.d_model, 2 * c.d_model)
        self.linear = nn.Linear(c.d_model, c.patch_size * c.patch_size * c.out_channels)

    def forward(self, x, cond):  # x: (batch, 256, d_model)
        shift, scale = self.ada(F.silu(cond))[:, None].chunk(2, dim=-1)
        x = self.linear(modulate(x, self.norm, shift, scale))  # (batch, 256, 32)
        p, side = self.c.patch_size, self.c.latent_size // self.c.patch_size
        x = x.view(-1, side, side, p, p, self.c.out_channels)  # (batch, row, col, dy, dx, channel)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (batch, channel, row, dy, col, dx)
        return x.reshape(-1, self.c.out_channels, side * p, side * p)  # (batch, 8, 32, 32)


# %% [scheduler-step] Guidance and DDIM step
# DDPM's noise schedule: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε, with β rising linearly over 1000 steps.
alphas_cumprod = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000), dim=0)  # ᾱ_t, t = 0…999


def guide(eps_class, eps_null, scale=4.0):
    """Classifier-free guidance: push the noise prediction away from the null class's."""
    return eps_null + scale * (eps_class - eps_null)


def ddim_step(x_t, eps, t, t_prev):
    """Deterministic DDIM (η = 0): estimate x₀ from the predicted noise, then re-noise it to t_prev."""
    a_t = alphas_cumprod[t]
    a_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)  # past t = 0: clean
    x0 = (x_t - (1 - a_t).sqrt() * eps) / a_t.sqrt()
    return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps, x0


# %% [model] Full model
class DiT(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.conditioning = Conditioning(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final = FinalLayer(c)

    def forward(self, x_t, t, class_ids):  # x_t: (batch, 4, 32, 32)
        cond = self.conditioning(t, class_ids)
        x = self.patchify(x_t)
        for block in self.blocks:
            x = block(x, cond)
        return self.final(x, cond)  # (batch, 8, 32, 32): noise, then variance


@torch.no_grad()
def sample(model, class_ids, generator, steps=50, guidance=4.0):
    """50 DDIM steps from pure noise, each running the class and the null class side by side."""
    c, n = model.c, len(class_ids)
    x = torch.randn(n, c.latent_channels, c.latent_size, c.latent_size, generator=generator)
    labels = torch.cat([class_ids, torch.full_like(class_ids, c.n_classes)])  # class, then null
    stride = 1000 // steps
    for t in range(1000 - stride, -1, -stride):  # 980, 960, …, 0
        eps_class, eps_null = model(torch.cat([x, x]), torch.full((2 * n,), t), labels)[:, : c.latent_channels].chunk(2)
        x, _ = ddim_step(x, guide(eps_class, eps_null, guidance), t, t - stride)
    return x  # image = vae.decode(x / 0.18215)


# latent = sample(DiT(), torch.tensor([207]), torch.Generator().manual_seed(0))  # 207: golden retriever
`;export{e as default};
