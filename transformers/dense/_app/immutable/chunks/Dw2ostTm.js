const e=`# Oasis 500M (Etched/oasis-500m), the Minecraft world model in the Oasis tab, written as plain PyTorch blocks.
# A ViT autoencoder turns each 640 × 360 frame into a 16 × 18 × 32 latent; a DiT whose blocks alternate attention
# within a frame and causal attention across frames predicts the next frame from the frames before it and the
# player's action at every frame. Sampling uses diffusion forcing: every frame carries its own noise level.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    image_height: int = 360
    image_width: int = 640
    vae_patch: int = 20  # 18 × 32 patches of 20 × 20 pixels
    latent_channels: int = 16
    vae_dim: int = 1024
    vae_heads: int = 16
    vae_encoder_depth: int = 6
    vae_decoder_depth: int = 12
    latent_scale: float = 0.07843137255  # latents are multiplied by this before the DiT sees them
    patch_size: int = 2  # the DiT sees 9 × 16 = 144 tokens per frame
    d_model: int = 1024
    n_heads: int = 16  # 64 dimensions per head
    n_layers: int = 16
    d_mlp: int = 4096
    action_dim: int = 25  # VPT's keyboard and mouse: hotbar, WASD, camera, jump, sneak, sprint, attack, use, …
    max_frames: int = 32
    noise_levels: int = 1000


# %% [vae-encode] ViT autoencoder: encode
def rotate_pairs(x, angles):
    """Turn adjacent pairs of the first 2 · angles.shape[-1] coordinates of x by their angle; keep the rest."""
    n = 2 * angles.shape[-1]
    cos, sin = angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1)
    head, rest = x[..., :n], x[..., n:]
    pairs = head.reshape(*head.shape[:-1], -1, 2)
    turned = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
    return torch.cat([head * cos + turned * sin, rest], dim=-1)


def axial_angles(rows, cols, n_freqs, max_freq):
    """2D RoPE on a grid: positions −1 … 1 along each axis, n_freqs frequencies from π to π·max_freq/2."""
    freqs = torch.linspace(1.0, max_freq / 2, n_freqs) * math.pi
    r = torch.linspace(-1, 1, rows)[:, None, None] * freqs  # (rows, 1, n)
    c = torch.linspace(-1, 1, cols)[None, :, None] * freqs  # (1, cols, n)
    return torch.cat([r.expand(rows, cols, -1), c.expand(rows, cols, -1)], dim=-1)  # (rows, cols, 2n)


class VAEBlock(nn.Module):
    """Pre-norm ViT block whose q and k turn with the patch's row and column."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads, self.grid = c.vae_heads, (c.image_height // c.vae_patch, c.image_width // c.vae_patch)
        self.norm1 = nn.LayerNorm(c.vae_dim, eps=1e-6)
        self.qkv = nn.Linear(c.vae_dim, 3 * c.vae_dim)
        self.proj = nn.Linear(c.vae_dim, c.vae_dim)
        self.norm2 = nn.LayerNorm(c.vae_dim, eps=1e-6)
        self.fc1 = nn.Linear(c.vae_dim, 4 * c.vae_dim)
        self.fc2 = nn.Linear(4 * c.vae_dim, c.vae_dim)
        head = c.vae_dim // c.vae_heads
        self.register_buffer("angles", axial_angles(*self.grid, head // 8, self.grid[0] * self.grid[1]).flatten(0, 1), persistent=False)

    def forward(self, x):  # (batch, 576 patches, 1024)
        b, n, d = x.shape
        q, k, v = (t.view(b, n, self.heads, -1).transpose(1, 2) for t in self.qkv(self.norm1(x)).chunk(3, dim=-1))
        q, k = rotate_pairs(q, self.angles), rotate_pairs(k, self.angles)  # 32 of 64 coordinates turn
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.patch = nn.Conv2d(3, c.vae_dim, kernel_size=c.vae_patch, stride=c.vae_patch)
        self.blocks = nn.ModuleList(VAEBlock(c) for _ in range(c.vae_encoder_depth))
        self.norm = nn.LayerNorm(c.vae_dim, eps=1e-6)
        self.moments = nn.Linear(c.vae_dim, 2 * c.latent_channels)  # mean and log-variance

    def forward(self, frames):  # (batch, 3, 360, 640) in [0, 1]
        x = self.patch(frames * 2 - 1).flatten(2).transpose(1, 2)  # (batch, 576, 1024)
        for block in self.blocks:
            x = block(x)
        mean = self.moments(self.norm(x)).chunk(2, dim=-1)[0]  # the mean; the sampler never draws from the posterior
        h, w = self.c.image_height // self.c.vae_patch, self.c.image_width // self.c.vae_patch
        return (mean * self.c.latent_scale).transpose(1, 2).reshape(-1, self.c.latent_channels, h, w)  # (batch, 16, 18, 32)


# %% [vae-decode] ViT autoencoder: decode
class Decoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.expand = nn.Linear(c.latent_channels, c.vae_dim)
        self.blocks = nn.ModuleList(VAEBlock(c) for _ in range(c.vae_decoder_depth))
        self.norm = nn.LayerNorm(c.vae_dim, eps=1e-6)
        self.pixels = nn.Linear(c.vae_dim, 3 * c.vae_patch**2)

    def forward(self, latent):  # (batch, 16, 18, 32), as the DiT makes it
        c, b = self.c, latent.shape[0]
        h, w, p = c.image_height // c.vae_patch, c.image_width // c.vae_patch, c.vae_patch
        x = self.expand(latent.flatten(2).transpose(1, 2) / c.latent_scale)
        for block in self.blocks:
            x = block(x)
        x = self.pixels(self.norm(x))  # (batch, 576, 3 · 20 · 20)
        x = x.view(b, h, w, 3, p, p).permute(0, 3, 1, 4, 2, 5)  # (batch, 3, row, dy, col, dx)
        return (x.reshape(b, 3, h * p, w * p) + 1) / 2  # pixels in [0, 1] before clamping


# %% [patch-embedding] Patch embedding
class PatchEmbedding(nn.Module):
    """Each frame's 16 × 18 × 32 latent → 9 × 16 tokens of 2 × 2 latent pixels. No position is added here:
    RoPE gives the row and column inside attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Conv2d(c.latent_channels, c.d_model, kernel_size=c.patch_size, stride=c.patch_size)

    def forward(self, latents):  # (batch, frames, 16, 18, 32)
        b, t = latents.shape[:2]
        x = self.proj(latents.flatten(0, 1))  # (batch · frames, d_model, 9, 16)
        return x.permute(0, 2, 3, 1).unflatten(0, (b, t))  # (batch, frames, 9, 16, d_model)


# %% [conditioning] Noise level and action, per frame
class Conditioning(nn.Module):
    """c = MLP(sin/cos of the frame's noise level) + W · the frame's action. Every frame has its own c, so one
    pass can denoise frames at different noise levels and steer each frame with its own keys and mouse."""

    def __init__(self, c: Config):
        super().__init__()
        self.mlp_in = nn.Linear(256, c.d_model)
        self.mlp_out = nn.Linear(c.d_model, c.d_model)
        self.action = nn.Linear(c.action_dim, c.d_model)

    @staticmethod
    def sinusoid(t, dim=256):
        """cos then sin of t at frequencies exp(−ln(10000) · i / 128)."""
        half = dim // 2
        angles = t.float()[:, None] * torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        return torch.cat([angles.cos(), angles.sin()], dim=-1)

    def forward(self, t, actions):  # t: (batch, frames) noise levels; actions: (batch, frames, 25)
        noise = self.mlp_out(F.silu(self.mlp_in(self.sinusoid(t.flatten())))).view(*t.shape, -1)
        return noise + self.action(actions)  # (batch, frames, d_model)


# %% [spatial-attention] Spatial attention
class SpatialAttention(nn.Module):
    """Within each frame, every token attends to all 144 tokens of the same frame. q and k turn with the token's
    row (first 32 coordinates) and column (last 32), positions spread over −1 … 1."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model)
        self.freqs = (c.d_model // c.n_heads // 4, 256)  # 16 frequencies per axis, up to π · 128

    def forward(self, x):  # (batch, frames, rows, cols, d_model)
        b, t, h, w, d = x.shape
        angles = axial_angles(h, w, *self.freqs).flatten(0, 1)  # (144, 32)
        q, k, v = (y.reshape(b * t, h * w, self.heads, -1).transpose(1, 2) for y in self.qkv(x).chunk(3, dim=-1))
        q, k = rotate_pairs(q, angles), rotate_pairs(k, angles)
        out = F.scaled_dot_product_attention(q, k, v)  # (batch · frames, heads, 144, 64)
        return self.out(out.transpose(1, 2).reshape(b, t, h, w, d))


# %% [temporal-attention] Temporal attention
class TemporalAttention(nn.Module):
    """At each of the 144 positions on its own, every frame attends to the same position in itself and the frames
    before it, never after. q and k turn with the frame's index in the window (RoPE, θ = 10000)."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # (batch, frames, rows, cols, d_model)
        b, t, h, w, d = x.shape
        q, k, v = (y.permute(0, 2, 3, 1, 4).reshape(b * h * w, t, self.heads, -1).transpose(1, 2) for y in self.qkv(x).chunk(3, dim=-1))
        head = q.shape[-1]
        angles = torch.arange(t, dtype=torch.float32)[:, None] / 10000 ** (torch.arange(0, head, 2).float() / head)
        q, k = rotate_pairs(q, angles), rotate_pairs(k, angles)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (batch · 144, heads, frames, 64)
        return self.out(out.transpose(1, 2).reshape(b, h, w, t, d).permute(0, 3, 1, 2, 4))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc1 = nn.Linear(c.d_model, c.d_mlp)
        self.fc2 = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


# %% [block] Spatio-temporal block
class Block(nn.Module):
    """A spatial half (attention within the frame, then an MLP) and a temporal half (causal attention across
    frames, then an MLP), each with its own adaLN-Zero: shift, scale and gate from the frame's c."""

    def __init__(self, c: Config):
        super().__init__()
        self.spatial_ada = nn.Linear(c.d_model, 6 * c.d_model)
        self.spatial = SpatialAttention(c)
        self.spatial_mlp = MLP(c)
        self.temporal_ada = nn.Linear(c.d_model, 6 * c.d_model)
        self.temporal = TemporalAttention(c)
        self.temporal_mlp = MLP(c)

    @staticmethod
    def half(x, cond, ada, attention, mlp):
        shift1, scale1, gate1, shift2, scale2, gate2 = (v[:, :, None, None] for v in ada(F.silu(cond)).chunk(6, dim=-1))
        norm = lambda y: F.layer_norm(y, y.shape[-1:], eps=1e-6)
        x = x + gate1 * attention(norm(x) * (1 + scale1) + shift1)
        return x + gate2 * mlp(norm(x) * (1 + scale2) + shift2)

    def forward(self, x, cond):  # x: (batch, frames, 9, 16, d_model); cond: (batch, frames, d_model)
        x = self.half(x, cond, self.spatial_ada, self.spatial, self.spatial_mlp)
        return self.half(x, cond, self.temporal_ada, self.temporal, self.temporal_mlp)


# %% [output] Output layer and unpatchify
class OutputLayer(nn.Module):
    """adaLN (shift and scale) from the frame's c, a Linear to 2 × 2 × 16 numbers per token, then the latent:
    the predicted velocity v of every frame."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.ada = nn.Linear(c.d_model, 2 * c.d_model)
        self.linear = nn.Linear(c.d_model, c.patch_size**2 * c.latent_channels)

    def forward(self, x, cond):
        shift, scale = (v[:, :, None, None] for v in self.ada(F.silu(cond)).chunk(2, dim=-1))
        x = self.linear(F.layer_norm(x, x.shape[-1:], eps=1e-6) * (1 + scale) + shift)  # (batch, frames, 9, 16, 64)
        b, t, h, w, _ = x.shape
        p = self.c.patch_size
        x = x.view(b, t, h, w, p, p, -1).permute(0, 1, 6, 2, 4, 3, 5)  # (batch, frames, 16, row, dy, col, dx)
        return x.reshape(b, t, -1, h * p, w * p)


# %% [scheduler-step] Diffusion forcing step
def sigmoid_schedule(levels=1000, start=-3.0, end=3.0):
    """ᾱ for noise levels 0 … 999: a sigmoid from 1 down to 0 (Jabri et al.), as cumulative products of β."""
    t = torch.linspace(0, levels, levels + 1, dtype=torch.float64) / levels
    v_start, v_end = torch.tensor(start).sigmoid(), torch.tensor(end).sigmoid()
    alpha_bar = (v_end - (t * (end - start) + start).sigmoid()) / (v_end - v_start)
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clip(0, 0.999)
    return torch.cumprod(1 - betas.float(), dim=0)


def ddim_step(x, v, t, t_next, alpha_bar, last):
    """The model predicts v = √ᾱ ε − √(1−ᾱ) x₀ for every frame. Only the newest frame moves to t_next (to t = 0 on
    the last step); the context frames stay as they are."""
    a = alpha_bar[t].view(*t.shape, 1, 1, 1)
    x0 = a.sqrt() * x - (1 - a).sqrt() * v
    eps = ((1 / a).sqrt() * x - x0) / (1 / a - 1).sqrt()
    a_next = torch.ones_like(a) if last else alpha_bar[t_next[:, -1:]].view(-1, 1, 1, 1, 1)
    newest = a_next.sqrt() * x0[:, -1:] + (1 - a_next).sqrt() * eps[:, -1:]
    return x0, torch.cat([x[:, :-1], newest], dim=1)


# %% [model] Full model and the diffusion-forcing rollout
class Oasis(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.conditioning = Conditioning(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = OutputLayer(c)

    def forward(self, latents, t, actions):  # latents: (batch, frames, 16, 18, 32); t: (batch, frames)
        cond = self.conditioning(t, actions)
        x = self.patchify(latents)
        for block in self.blocks:
            x = block(x, cond)
        return self.output(x, cond)  # v for every frame


@torch.no_grad()
def rollout(model, prompt, actions, frames=32, steps=10, context_level=15, generator=None):
    """Diffusion forcing, one new frame at a time: append pure noise (clipped at ±20) and denoise it in \`steps\`
    DDIM steps from noise level 999 while the frames before it are told they sit at level 14. prompt: the encoded
    first frame(s), (batch, n, 16, 18, 32); actions: (batch, frames, 25), action i steers frame i."""
    alpha_bar = sigmoid_schedule(model.c.noise_levels)
    levels = torch.linspace(-1, model.c.noise_levels - 1, steps + 1)  # −1, 99, 199, …, 999
    x = prompt
    for i in range(prompt.shape[1], frames):
        b = x.shape[0]
        x = torch.cat([x, torch.randn((b, 1, *x.shape[-3:]), generator=generator).clamp(-20, 20)], dim=1)
        first = max(0, i + 1 - model.c.max_frames)  # a sliding window of at most 32 frames
        for k in range(steps, 0, -1):
            context = torch.full((b, i), context_level - 1, dtype=torch.long)
            t = torch.cat([context, torch.full((b, 1), int(levels[k]), dtype=torch.long)], dim=1)[:, first:]
            t_next = torch.cat([context, torch.full((b, 1), max(int(levels[k - 1]), 0), dtype=torch.long)], dim=1)[:, first:]
            v = model(x[:, first:], t, actions[:, first : i + 1])
            _, window = ddim_step(x[:, first:], v, t, t_next, alpha_bar, last=k == 1)
            x[:, -1:] = window[:, -1:]
    return x  # every frame's latent; Decoder turns each into a 640 × 360 frame


# vae = Encoder(Config()), Decoder(Config()); prompt = encoder(first_frame)[:, None]
# video = decoder(rollout(Oasis(), prompt, actions).flatten(0, 1)).clamp(0, 1)
`;export{e as default};
