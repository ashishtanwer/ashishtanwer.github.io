const n=`# PixArt-α XL/2 512 px, the text-to-image diffusion transformer in the PixArt-α tab, written as plain PyTorch blocks.
# T5-XXL reads the prompt; 28 blocks predict the noise in a 4 × 64 × 64 VAE latent, reading the prompt through
# cross-attention and the timestep through adaLN-single; DPM-Solver++ takes 20 steps and the VAE decodes 512 × 512 px.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # T5 v1.1 XXL, encoder only
    vocab: int = 32128
    t5_dim: int = 4096
    t5_heads: int = 64  # 64 dimensions per head
    t5_head_dim: int = 64
    t5_ff: int = 10240
    t5_layers: int = 24
    t5_buckets: int = 32
    t5_max_distance: int = 128
    max_tokens: int = 120  # prompts are padded to 120 tokens; padding is masked
    # The diffusion transformer
    latent_channels: int = 4
    latent_size: int = 64  # 512 px / the VAE's factor of 8
    patch_size: int = 2  # 32 × 32 = 1024 patches of 2 × 2 × 4 numbers
    d_model: int = 1152
    n_heads: int = 16  # 1152 / 16 = 72 dimensions per head
    n_layers: int = 28
    d_mlp: int = 4608  # 4 × d_model
    out_channels: int = 8  # predicted noise (4) and a learned variance the sampler does not use (4)
    frequency_dim: int = 256  # cos and sin of t at 128 frequencies
    # The VAE's decoder (Stable Diffusion's 8× KL autoencoder)
    vae_channels: tuple = (128, 256, 512, 512)
    vae_layers: int = 3  # residual blocks per decoder stage
    vae_groups: int = 32
    vae_scaling: float = 0.18215


# %% [text-encoder] T5-XXL text encoder
def relative_bucket(relative, buckets=32, max_distance=128):
    """T5's bidirectional position buckets: one half for keys to the right, one for the left; exact up to 7
    tokens away, then log-spaced up to 128."""
    half = buckets // 2
    exact = half // 2
    distance = relative.abs()
    far = exact + (torch.log(distance.clamp(min=1).float() / exact) / math.log(max_distance / exact) * (half - exact))
    return (relative > 0).long() * half + torch.where(distance < exact, distance, far.long().clamp(max=half - 1))


class RMSNorm(nn.Module):
    """T5's layer norm: scale by the root mean square; no mean and no bias."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class T5Layer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        inner = c.t5_heads * c.t5_head_dim
        self.heads = c.t5_heads
        self.norm1 = RMSNorm(c.t5_dim)
        self.q, self.k, self.v = (nn.Linear(c.t5_dim, inner, bias=False) for _ in range(3))
        self.o = nn.Linear(inner, c.t5_dim, bias=False)
        self.norm2 = RMSNorm(c.t5_dim)
        self.wi_0 = nn.Linear(c.t5_dim, c.t5_ff, bias=False)  # GELU branch
        self.wi_1 = nn.Linear(c.t5_dim, c.t5_ff, bias=False)  # linear branch
        self.wo = nn.Linear(c.t5_ff, c.t5_dim, bias=False)

    def forward(self, x, bias):  # bias: relative positions + padding mask, (batch, heads, n, n)
        h = self.norm1(x)
        q, k, v = (p(h).unflatten(-1, (self.heads, -1)).transpose(1, 2) for p in (self.q, self.k, self.v))
        weights = torch.softmax(q @ k.transpose(-1, -2) + bias, dim=-1)  # no 1/√d: T5 folds it into its weights
        x = x + self.o((weights @ v).transpose(1, 2).flatten(2))
        h = self.norm2(x)
        return x + self.wo(F.gelu(self.wi_0(h), approximate="tanh") * self.wi_1(h))  # gated GELU


class T5Encoder(nn.Module):
    """Token ids → one 4096-wide vector per token, every token reading every other (no causal mask)."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.t5_dim)
        self.relative = nn.Embedding(c.t5_buckets, c.t5_heads)  # learned in layer 1, reused by every layer
        self.layers = nn.ModuleList(T5Layer(c) for _ in range(c.t5_layers))
        self.norm = RMSNorm(c.t5_dim)

    def forward(self, ids, mask):  # ids, mask: (batch, 120); mask is 1 on the prompt, 0 on padding
        n = ids.shape[1]
        position = torch.arange(n)
        buckets = relative_bucket(position[None] - position[:, None], self.c.t5_buckets, self.c.t5_max_distance)
        bias = self.relative(buckets).permute(2, 0, 1)[None]  # (1, heads, n, n), key minus query
        bias = bias + (1 - mask[:, None, None].float()) * torch.finfo(torch.float32).min
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x, bias)
        return self.norm(x)


# %% [caption-projection] Caption projection
class CaptionProjection(nn.Module):
    """T5's 4096-wide token vectors → y, 1152 wide: the keys and values every block's cross-attention reads."""

    def __init__(self, c: Config):
        super().__init__()
        self.linear_1 = nn.Linear(c.t5_dim, c.d_model)
        self.linear_2 = nn.Linear(c.d_model, c.d_model)

    def forward(self, text):  # (batch, 120, 4096)
        return self.linear_2(F.gelu(self.linear_1(text), approximate="tanh"))


# %% [patchify] Patch embedding
def sincos(d, positions):
    """d/2 sines then d/2 cosines of each position at frequencies 10000^(−2i/d)."""
    frequencies = 10000 ** (-torch.arange(d // 2, dtype=torch.float64) / (d / 2))
    angles = positions[:, None].double() * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class PatchEmbedding(nn.Module):
    """Cut the latent into 2 × 2 patches, project each to 1152, add a fixed 2D sin-cos position."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Conv2d(c.latent_channels, c.d_model, kernel_size=c.patch_size, stride=c.patch_size)
        side = c.latent_size // c.patch_size
        rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        # First half of each vector encodes the patch's column, second half its row; not learned.
        position = torch.cat([sincos(c.d_model // 2, cols.flatten()), sincos(c.d_model // 2, rows.flatten())], dim=1)
        self.register_buffer("position", position.float(), persistent=False)  # (1024, 1152)

    def forward(self, latent):  # (batch, 4, 64, 64)
        return self.proj(latent).flatten(2).transpose(1, 2) + self.position  # (batch, 1024, 1152), row by row


# %% [adaln-single] adaLN-single
def sinusoid(t, d=256):
    """cos then sin of t at d/2 frequencies from 1 down to 1/10000."""
    half = d // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    angles = t[:, None].float() * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class AdaLNSingle(nn.Module):
    """One timestep MLP and one Linear for all 28 blocks (DiT gives every block its own 1152 → 6 × 1152 Linear).
    t₀ = MLP(sinusoid(t)); shared = Linear(SiLU(t₀)), six 1152-wide vectors; each block adds its own 6 × 1152 table."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.mlp_in = nn.Linear(c.frequency_dim, c.d_model)
        self.mlp_out = nn.Linear(c.d_model, c.d_model)
        self.linear = nn.Linear(c.d_model, 6 * c.d_model)

    def forward(self, t):  # t: (batch,) timesteps 0…999
        t0 = self.mlp_out(F.silu(self.mlp_in(sinusoid(t, self.frequency_dim))))
        return self.linear(F.silu(t0)), t0  # shared: (batch, 6 · 1152), t₀: (batch, 1152)


def modulation(shared, table):
    """Block-specific shift β, scale γ and gate α: the shared vectors plus the block's learned table (6, 1152)."""
    return (table[None] + shared.unflatten(1, (6, -1))).chunk(6, dim=1)  # six (batch, 1, 1152): β₁ γ₁ α₁ β₂ γ₂ α₂


def layer_norm(x):
    return F.layer_norm(x, x.shape[-1:], eps=1e-6)  # no learned scale or shift: adaLN supplies them


# %% [self-attention] Self-attention over the patches
class Attention(nn.Module):
    """Multi-head attention with biases: queries from x, keys and values from \`context\` (x itself by default)."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.n_heads
        self.q, self.k, self.v, self.out = (nn.Linear(c.d_model, c.d_model) for _ in range(4))

    def forward(self, x, context=None, bias=None):
        context = x if context is None else context
        q = self.q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)  # (batch, 16, n, 72)
        k = self.k(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v = self.v(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if bias is not None:
            scores = scores + bias
        return self.out((scores.softmax(dim=-1) @ v).transpose(1, 2).flatten(2))


# %% [cross-attention] Cross-attention to the prompt
def cross_attention(attention, x, y, mask):
    """Every patch's query reads the prompt's tokens y. Padding slots get −10000, so their weight is 0."""
    bias = (1 - mask.float())[:, None, None, :] * -10000.0  # (batch, 1, 1, 120)
    return attention(x, context=y, bias=bias)


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] PixArt block
class Block(nn.Module):
    """adaLN self-attention, then cross-attention to the prompt, then the adaLN MLP."""

    def __init__(self, c: Config):
        super().__init__()
        self.attention = Attention(c)
        self.cross = Attention(c)
        self.mlp = MLP(c)
        self.table = nn.Parameter(torch.zeros(6, c.d_model))  # this block's own offsets to the shared vectors

    def forward(self, x, shared, y, mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation(shared, self.table)
        x = x + gate1 * self.attention(layer_norm(x) * (1 + scale1) + shift1)
        x = x + cross_attention(self.cross, x, y, mask)  # no norm and no gate: the patches read the prompt as they are
        return x + gate2 * self.mlp(layer_norm(x) * (1 + scale2) + shift2)


# %% [final-layer] Final layer and unpatchify
class FinalLayer(nn.Module):
    """Shift and scale from t₀ plus a learned 2 × 1152 table, then 1152 → 2 · 2 · 8 per patch, back into a grid."""

    def __init__(self, c: Config):
        super().__init__()
        self.p, self.out_channels = c.patch_size, c.out_channels
        self.table = nn.Parameter(torch.zeros(2, c.d_model))
        self.linear = nn.Linear(c.d_model, c.patch_size**2 * c.out_channels)

    def forward(self, x, t0, side):
        shift, scale = (self.table[None] + t0[:, None]).chunk(2, dim=1)
        x = self.linear(layer_norm(x) * (1 + scale) + shift)  # (batch, 1024, 32)
        x = x.view(-1, side, side, self.p, self.p, self.out_channels).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(-1, self.out_channels, side * self.p, side * self.p)  # (batch, 8, 64, 64)


# %% [scheduler-step] Guidance and the DPM-Solver++ step
def schedule(steps, train_steps=1000):
    """20 timesteps from 999 down to 49, and α_t = √ᾱ_t, σ_t = √(1 − ᾱ_t) for the linear β schedule; the step after
    the last one has σ = 0, a clean latent."""
    alphas_cumprod = torch.cumprod(1 - torch.linspace(1e-4, 0.02, train_steps, dtype=torch.float32), dim=0)
    t = torch.linspace(0, train_steps - 1, steps + 1).round().flip(0)[:-1].long()
    alpha = torch.cat([alphas_cumprod[t].sqrt(), torch.ones(1)])
    sigma = torch.cat([(1 - alphas_cumprod[t]).sqrt(), torch.zeros(1)])
    return t, alpha, sigma


def guided_noise(eps_unconditional, eps_conditional, scale=4.5):
    return eps_unconditional + scale * (eps_conditional - eps_unconditional)


def dpm_solver_step(x, eps, i, alpha, sigma, previous_x0=None):
    """x̂₀ = (x − σ ε̂) / α, then one DPM-Solver++ step in λ = log(α/σ). With the previous step's x̂₀ it is second
    order (midpoint); the first and the last step are first order."""
    x0 = (x - sigma[i] * eps) / alpha[i]
    if sigma[i + 1] == 0:  # the last step lands on x̂₀
        return x0, x0
    lam = torch.log(alpha / sigma)
    h = lam[i + 1] - lam[i]
    x_next = (sigma[i + 1] / sigma[i]) * x - alpha[i + 1] * (torch.exp(-h) - 1) * x0
    if previous_x0 is not None:
        r = (lam[i] - lam[i - 1]) / h
        x_next = x_next - 0.5 * alpha[i + 1] * (torch.exp(-h) - 1) * (x0 - previous_x0) / r
    return x_next, x0


# %% [vae-decode] VAE decoder
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in, eps=1e-6)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out, eps=1e-6)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return self.skip(x) + h


class PixelAttention(nn.Module):
    """One head of self-attention over the 64 × 64 positions, in the decoder's middle."""

    def __init__(self, c, groups):
        super().__init__()
        self.norm = nn.GroupNorm(groups, c, eps=1e-6)
        self.q, self.k, self.v, self.out = (nn.Linear(c, c) for _ in range(4))

    def forward(self, x):
        b, c, height, width = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)  # (batch, positions, c)
        weights = torch.softmax(self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(c), dim=-1)
        return x + self.out(weights @ self.v(h)).transpose(1, 2).view(b, c, height, width)


class VAEDecoder(nn.Module):
    """4 × 64 × 64 latent → 3 × 512 × 512 image: three 2× upsamplings between stages of residual blocks."""

    def __init__(self, c: Config):
        super().__init__()
        widths, groups = list(reversed(c.vae_channels)), c.vae_groups  # 512, 512, 256, 128
        self.scaling = c.vae_scaling
        self.post_quant = nn.Conv2d(c.latent_channels, c.latent_channels, 1)
        self.conv_in = nn.Conv2d(c.latent_channels, widths[0], 3, padding=1)
        self.mid = nn.Sequential(
            ResBlock(widths[0], widths[0], groups), PixelAttention(widths[0], groups), ResBlock(widths[0], widths[0], groups)
        )
        self.stages = nn.ModuleList()
        for i, width in enumerate(widths):
            stage = nn.Module()
            c_in = widths[max(0, i - 1)]
            stage.blocks = nn.Sequential(*(ResBlock(c_in if j == 0 else width, width, groups) for j in range(c.vae_layers)))
            stage.upsample = nn.Conv2d(width, width, 3, padding=1) if i < len(widths) - 1 else None
            self.stages.append(stage)
        self.norm = nn.GroupNorm(groups, widths[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(widths[-1], 3, 3, padding=1)

    def forward(self, latent):
        x = self.mid(self.conv_in(self.post_quant(latent / self.scaling)))  # undo the latent's 0.18215 scaling
        for stage in self.stages:
            x = stage.blocks(x)
            if stage.upsample is not None:
                x = stage.upsample(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        image = self.conv_out(F.silu(self.norm(x)))  # (batch, 3, 512, 512) in [−1, 1]
        return (image / 2 + 0.5).clamp(0, 1)


# %% [model] Full model
class PixArt(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.caption = CaptionProjection(c)
        self.patchify = PatchEmbedding(c)
        self.adaln = AdaLNSingle(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final = FinalLayer(c)

    def forward(self, latent, t, text, mask):  # text: T5 output (batch, 120, 4096); mask: (batch, 120)
        x = self.patchify(latent)
        shared, t0 = self.adaln(t)
        y = self.caption(text)
        for block in self.blocks:
            x = block(x, shared, y, mask)
        return self.final(x, t0, latent.shape[-1] // self.c.patch_size)


@torch.no_grad()
def generate(model, vae, text, mask, null_text, null_mask, steps=20, scale=4.5, generator=None):
    """Text embeddings → image: classifier-free guidance with the empty prompt, 20 DPM-Solver++ steps, VAE decode."""
    c = model.c
    latent = torch.randn(1, c.latent_channels, c.latent_size, c.latent_size, generator=generator)
    t, alpha, sigma = schedule(steps)
    texts, masks = torch.cat([null_text, text]), torch.cat([null_mask, mask])
    previous = None
    for i in range(steps):
        out = model(torch.cat([latent, latent]), t[i].expand(2), texts, masks)[:, : c.latent_channels]
        eps = guided_noise(*out.chunk(2), scale)
        latent, x0 = dpm_solver_step(latent, eps, i, alpha, sigma, previous if 0 < i < steps - 1 else None)
        previous = x0
    return latent, vae(latent)
`;export{n as default};
