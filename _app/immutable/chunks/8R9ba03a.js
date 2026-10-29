const e=`# Stable Audio Open 1.0, the audio diffusion transformer in the Stable Audio tab, written as plain PyTorch blocks.
# T5-base reads the prompt and two numbers say where the clip starts and how long it is; 24 blocks denoise a
# 64 × 1024 audio latent (47.6 s at 21.5 latent frames a second) predicting v, an SDE DPM-Solver++ takes the steps
# on EDM sigmas, and the Oobleck VAE decodes 2048 stereo samples per latent frame at 44.1 kHz.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # T5-base encoder (ReLU feed-forward)
    vocab: int = 32128
    t5_dim: int = 768
    t5_heads: int = 12
    t5_head_dim: int = 64
    t5_ff: int = 3072
    t5_layers: int = 12
    t5_buckets: int = 32
    t5_max_distance: int = 128
    # Seconds conditioning
    seconds_max: float = 512.0
    number_dim: int = 256  # learned Fourier features of a number in [0, 1]
    # The diffusion transformer
    latent_channels: int = 64
    latent_frames: int = 1024
    d_model: int = 1536
    n_heads: int = 24  # 24 heads of 64
    kv_heads: int = 12  # cross-attention shares each key/value head between two query heads
    head_dim: int = 64
    n_layers: int = 24
    d_mlp: int = 6144  # SwiGLU: 2 × 6144 before the gate
    cond_dim: int = 768
    time_dim: int = 256
    rope_dim: int = 32  # RoPE turns the first 32 of each head's 64 coordinates
    # EDM noise levels
    sigma_min: float = 0.3
    sigma_max: float = 500.0
    sigma_data: float = 1.0
    # The Oobleck VAE's decoder
    vae_channels: int = 128
    vae_multiples: tuple = (1, 2, 4, 8, 16)
    vae_strides: tuple = (2, 4, 4, 8, 8)  # 2048 audio samples per latent frame
    audio_channels: int = 2


# %% [text-encoder] T5-base text encoder
def relative_bucket(relative, buckets=32, max_distance=128):
    """T5's bidirectional position buckets: exact up to 7 tokens away, then log-spaced up to 128."""
    half = buckets // 2
    exact = half // 2
    distance = relative.abs()
    far = exact + (torch.log(distance.clamp(min=1).float() / exact) / math.log(max_distance / exact) * (half - exact))
    return (relative > 0).long() * half + torch.where(distance < exact, distance, far.long().clamp(max=half - 1))


class RMSNorm(nn.Module):
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
        self.wi = nn.Linear(c.t5_dim, c.t5_ff, bias=False)
        self.wo = nn.Linear(c.t5_ff, c.t5_dim, bias=False)

    def forward(self, x, bias):
        h = self.norm1(x)
        q, k, v = (p(h).unflatten(-1, (self.heads, -1)).transpose(1, 2) for p in (self.q, self.k, self.v))
        x = x + self.o((torch.softmax(q @ k.transpose(-1, -2) + bias, dim=-1) @ v).transpose(1, 2).flatten(2))
        return x + self.wo(F.relu(self.wi(self.norm2(x))))


class T5Encoder(nn.Module):
    """Token ids → one 768-wide vector per token; padding slots are then set to zero, as the pipeline does."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.t5_dim)
        self.relative = nn.Embedding(c.t5_buckets, c.t5_heads)
        self.layers = nn.ModuleList(T5Layer(c) for _ in range(c.t5_layers))
        self.norm = RMSNorm(c.t5_dim)

    def forward(self, ids, mask):
        position = torch.arange(ids.shape[1])
        buckets = relative_bucket(position[None] - position[:, None], self.c.t5_buckets, self.c.t5_max_distance)
        bias = self.relative(buckets).permute(2, 0, 1)[None]
        bias = bias + (1 - mask[:, None, None].float()) * torch.finfo(torch.float32).min
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x, bias)
        return self.norm(x) * mask[..., None]  # padding → 0: it still takes part in cross-attention, as zeros


# %% [seconds] Seconds conditioning
class NumberEmbedding(nn.Module):
    """A number of seconds → 768: clamp to [0, 512], scale to [0, 1], learned Fourier features, a Linear."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.frequencies = nn.Parameter(torch.zeros(c.number_dim // 2))  # learned
        self.linear = nn.Linear(c.number_dim + 1, c.cond_dim)

    def forward(self, seconds):  # (batch,)
        t = (seconds.clamp(0, self.c.seconds_max) / self.c.seconds_max)[:, None]
        angles = 2 * math.pi * t * self.frequencies
        return self.linear(torch.cat([t, angles.sin(), angles.cos()], dim=-1))[:, None]  # (batch, 1, 768)


class Seconds(nn.Module):
    """Two separate embeddings: where the clip starts in the source audio, and how long it is in total."""

    def __init__(self, c: Config):
        super().__init__()
        self.start, self.total = NumberEmbedding(c), NumberEmbedding(c)

    def forward(self, seconds_start, seconds_total):
        return self.start(seconds_start), self.total(seconds_total)


# %% [conditioning] Prompt tokens and the global token
class Conditioning(nn.Module):
    """Cross-attention reads 130 tokens: the 128 prompt slots, then seconds_start and seconds_total. The global token
    puts the two numbers side by side (1536) and adds the timestep's Fourier embedding; it is prepended to the latent
    frames, so self-attention sees it as frame 0."""

    def __init__(self, c: Config):
        super().__init__()
        self.cross_in = nn.Linear(c.cond_dim, c.cond_dim, bias=False)
        self.cross_out = nn.Linear(c.cond_dim, c.cond_dim, bias=False)
        self.global_in = nn.Linear(2 * c.cond_dim, c.d_model, bias=False)
        self.global_out = nn.Linear(c.d_model, c.d_model, bias=False)
        self.time_frequencies = nn.Parameter(torch.zeros(c.time_dim // 2), requires_grad=False)  # fixed random
        self.time_in = nn.Linear(c.time_dim, c.d_model)
        self.time_out = nn.Linear(c.d_model, c.d_model)

    def forward(self, text, start, total, t):  # text: (batch, 128, 768); start, total: (batch, 1, 768); t: (batch,)
        tokens = torch.cat([text, start, total], dim=1)
        cross = self.cross_out(F.silu(self.cross_in(tokens)))  # (batch, 130, 768)
        angles = 2 * math.pi * t[:, None] * self.time_frequencies
        time = self.time_out(F.silu(self.time_in(torch.cat([angles.cos(), angles.sin()], dim=-1))))
        global_token = self.global_out(F.silu(self.global_in(torch.cat([start, total], dim=-1)))) + time[:, None]
        return cross, global_token  # (batch, 1, 1536)


# %% [self-attention] Self-attention over the latent frames
def rope(x, rotate=32, theta=10000.0):
    """Rotate the first \`rotate\` coordinates of every head by position × θ^(−2i/rotate); leave the rest."""
    n = x.shape[-2]
    angles = torch.arange(n, dtype=torch.float32)[:, None] * theta ** (-torch.arange(0, rotate, 2) / rotate)
    cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
    turned, kept = x[..., :rotate], x[..., rotate:]
    half = rotate // 2
    turned = turned * cos + torch.cat([-turned[..., half:], turned[..., :half]], -1) * sin
    return torch.cat([turned, kept], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q, self.k, self.v, self.out = (nn.Linear(c.d_model, c.d_model, bias=False) for _ in range(4))

    def forward(self, x):  # x: (batch, 1 + frames, 1536), the global token first
        q, k, v = (p(x).unflatten(-1, (self.c.n_heads, -1)).transpose(1, 2) for p in (self.q, self.k, self.v))
        q, k = rope(q, self.c.rope_dim), rope(k, self.c.rope_dim)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.c.head_dim), dim=-1)
        return self.out((weights @ v).transpose(1, 2).flatten(2))


# %% [cross-attention] Cross-attention to the prompt and the seconds
class CrossAttention(nn.Module):
    """24 query heads, 12 key/value heads (grouped-query): each key/value head serves two query heads. No RoPE and
    no mask: the zeroed padding slots get keys of 0, so each takes the weight exp(0) / Σ."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.cond_dim, c.kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.cond_dim, c.kv_heads * c.head_dim, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cross):
        c = self.c
        q = self.q(x).unflatten(-1, (c.n_heads, -1)).transpose(1, 2)
        k, v = (p(cross).unflatten(-1, (c.kv_heads, -1)).transpose(1, 2) for p in (self.k, self.v))
        k, v = (t.repeat_interleave(c.n_heads // c.kv_heads, dim=1) for t in (k, v))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(c.head_dim), dim=-1)
        return self.out((weights @ v).transpose(1, 2).flatten(2))


# %% [mlp] SwiGLU feed-forward
class SwiGLU(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, 2 * c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(value * F.silu(gate))


# %% [block] Stable Audio block
class Block(nn.Module):
    """Pre-norm, with learned LayerNorms and no adaLN: the timestep reaches the blocks only through the global token."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(c.d_model, eps=1e-5) for _ in range(3))
        self.attention = SelfAttention(c)
        self.cross = CrossAttention(c)
        self.mlp = SwiGLU(c)

    def forward(self, x, cross):
        x = x + self.attention(self.norm1(x))
        x = x + self.cross(self.norm2(x), cross)
        return x + self.mlp(self.norm3(x))


# %% [model-io] Latent frames in and out
class LatentIO(nn.Module):
    """64 latent channels per frame → 1536 wide, with a residual 1 × 1 conv on each side."""

    def __init__(self, c: Config):
        super().__init__()
        self.pre = nn.Conv1d(c.latent_channels, c.latent_channels, 1, bias=False)
        self.proj_in = nn.Linear(c.latent_channels, c.d_model, bias=False)
        self.proj_out = nn.Linear(c.d_model, c.latent_channels, bias=False)
        self.post = nn.Conv1d(c.latent_channels, c.latent_channels, 1, bias=False)

    def embed(self, latent, global_token):  # latent: (batch, 64, frames)
        frames = self.proj_in((self.pre(latent) + latent).transpose(1, 2))
        return torch.cat([global_token, frames], dim=1)  # (batch, 1 + frames, 1536)

    def unembed(self, x):
        out = self.proj_out(x).transpose(1, 2)[:, :, 1:]  # drop the global token's row
        return self.post(out) + out  # (batch, 64, frames)


# %% [sampler-step] EDM preconditioning and the SDE DPM-Solver++ step
def sigmas(steps, c: Config):
    """Log-spaced noise levels from 500 down to 0.3, then 0."""
    s = torch.linspace(math.log(c.sigma_min), math.log(c.sigma_max), steps).exp().flip(0)
    return torch.cat([s, torch.zeros(1)])


def precondition(x, sigma, c: Config):
    """The transformer sees x / √(σ² + σ_d²) and t = atan(σ) · 2/π, a number in (0, 1)."""
    return x / math.sqrt(sigma**2 + c.sigma_data**2), torch.atan(torch.tensor([sigma])) * 2 / math.pi


def denoised(x, v, sigma, c: Config):
    """x̂₀ from the predicted v: c_skip x + c_out v, with c_skip = σ_d² / (σ² + σ_d²) and c_out = −σ σ_d / √(σ² + σ_d²)."""
    return c.sigma_data**2 / (sigma**2 + c.sigma_data**2) * x - sigma * c.sigma_data / math.sqrt(sigma**2 + c.sigma_data**2) * v


def sde_step(x, x0, i, s, noise, previous_x0=None):
    """One step from σ_i to σ_{i+1} with λ = −log σ, h = λ_{i+1} − λ_i: the deterministic part of DPM-Solver++ shrunk
    by e^{−h}, plus fresh noise of size σ_{i+1} √(1 − e^{−2h}). Second order (midpoint) between the first and last steps."""
    if s[i + 1] == 0:
        return x0
    h = torch.log(s[i]) - torch.log(s[i + 1])
    x_next = s[i + 1] / s[i] * torch.exp(-h) * x + (1 - torch.exp(-2 * h)) * x0
    if previous_x0 is not None:
        r = (torch.log(s[i - 1]) - torch.log(s[i])) / h
        x_next = x_next + 0.5 * (1 - torch.exp(-2 * h)) * (x0 - previous_x0) / r
    return x_next + s[i + 1] * torch.sqrt(1 - torch.exp(-2 * h)) * noise


# %% [vae-decode] Oobleck VAE decoder
def weight_norm(g, v):
    """Weight normalisation: each output filter's direction v scaled to length g."""
    return g * v / v.flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1))


class Snake(nn.Module):
    """x + sin²(αx) / β per channel, with α and β stored as logs."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        return x + (self.beta.exp() + 1e-9).reciprocal() * torch.sin(self.alpha.exp() * x).pow(2)


class WNConv(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation=1, padding=0, bias=True, transpose=False, stride=1):
        super().__init__()
        shape = (c_in, c_out, kernel) if transpose else (c_out, c_in, kernel)
        self.g = nn.Parameter(torch.ones(shape[0], 1, 1))
        self.v = nn.Parameter(torch.randn(shape))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.transpose, self.stride, self.dilation, self.padding = transpose, stride, dilation, padding

    def forward(self, x):
        w = weight_norm(self.g, self.v)
        if self.transpose:
            return F.conv_transpose1d(x, w, self.bias, stride=self.stride, padding=self.padding)
        return F.conv1d(x, w, self.bias, dilation=self.dilation, padding=self.padding)


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.snake1, self.snake2 = Snake(channels), Snake(channels)
        self.conv1 = WNConv(channels, channels, 7, dilation=dilation, padding=3 * dilation)
        self.conv2 = WNConv(channels, channels, 1)

    def forward(self, x):
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class DecoderBlock(nn.Module):
    """Snake, a transposed conv that upsamples by \`stride\`, and three residual units dilated 1, 3 and 9."""

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.snake = Snake(c_in)
        self.up = WNConv(c_in, c_out, 2 * stride, transpose=True, stride=stride, padding=math.ceil(stride / 2))
        self.units = nn.Sequential(*(ResidualUnit(c_out, d) for d in (1, 3, 9)))

    def forward(self, x):
        return self.units(self.up(self.snake(x)))


class OobleckDecoder(nn.Module):
    """64 × frames → 2 × (2048 · frames): a 44.1 kHz stereo waveform."""

    def __init__(self, c: Config):
        super().__init__()
        widths = [c.vae_channels * m for m in (1, *c.vae_multiples)]  # 128, 128, 256, 512, 1024, 2048
        self.conv_in = WNConv(c.latent_channels, widths[-1], 7, padding=3)
        strides = list(reversed(c.vae_strides))  # 8, 8, 4, 4, 2
        self.blocks = nn.Sequential(
            *(DecoderBlock(widths[len(strides) - i], widths[len(strides) - i - 1], s) for i, s in enumerate(strides))
        )
        self.snake = Snake(c.vae_channels)
        self.conv_out = WNConv(c.vae_channels, c.audio_channels, 7, padding=3, bias=False)

    def forward(self, latent):
        return self.conv_out(self.snake(self.blocks(self.conv_in(latent))))


# %% [model] Full model
class StableAudioDiT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.conditioning = Conditioning(c)
        self.io = LatentIO(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))

    def forward(self, latent, t, text, start, total):  # start, total: (batch, 1, 768) seconds embeddings
        cross, global_token = self.conditioning(text, start, total, t)
        x = self.io.embed(latent, global_token)
        for block in self.blocks:
            x = block(x, cross)
        return self.io.unembed(x)  # v̂: (batch, 64, frames)


@torch.no_grad()
def generate(model, seconds, vae, text, null_text, seconds_start, seconds_total, noises, steps=100, scale=7.0, generator=None):
    """\`noises[i]\` is the fresh noise of step i (the pipeline draws it from a Brownian tree). \`null_text\` is the negative
    prompt's T5 output; both passes share the seconds."""
    c = model.c
    latent = torch.randn(1, c.latent_channels, c.latent_frames, generator=generator) * math.sqrt(c.sigma_max**2 + 1)
    s = sigmas(steps, c)
    start, total = seconds(seconds_start, seconds_total)
    texts, starts, totals = torch.cat([null_text, text]), torch.cat([start, start]), torch.cat([total, total])
    previous = None
    for i in range(steps):
        scaled, t = precondition(latent, s[i].item(), c)
        v_null, v_prompt = model(torch.cat([scaled, scaled]), t.expand(2), texts, starts, totals).chunk(2)
        v = v_null + scale * (v_prompt - v_null)
        x0 = denoised(latent, v, s[i].item(), c)
        latent = sde_step(latent, x0, i, s, noises[i], previous if 0 < i < steps - 1 else None)
        previous = x0
    return latent, vae(latent)
`;export{e as default};
