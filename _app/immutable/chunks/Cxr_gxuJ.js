const e=`# Multi-head latent attention as in DeepSeek-V2-Lite, the model in the Attention Explainer's MLA
# tab. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 2048
    n_heads: int = 16
    d_content: int = 128  # per-head query and key dimensions that carry no position
    d_rotary: int = 64  # per-head query dimensions RoPE rotates; the one 64-D rotary key is shared
    d_value: int = 128
    d_latent: int = 512  # c_KV; with the rotary key, 576 numbers per token are all the cache keeps
    eps: float = 1e-6
    rope_theta: float = 10000.0
    # YaRN stretches RoPE from the 4,096 positions it was trained on to 163,840.
    rope_factor: float = 40.0
    rope_original_positions: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 0.707


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [query] Query
class Query(nn.Module):
    """W_Q gives every head a 128-D content query and a 64-D query for RoPE. V2-Lite does not
    compress its queries; the larger DeepSeek-V2 does, through a latent of its own."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.proj = nn.Linear(c.d_model, c.n_heads * (c.d_content + c.d_rotary), bias=False)  # 2048 → 3072

    def forward(self, x):  # x: (batch, seq, d_model), the input after RMSNorm
        b, t, _ = x.shape
        q = self.proj(x).view(b, t, self.c.n_heads, -1).transpose(1, 2)  # (batch, 16, seq, 192)
        return q.split([self.c.d_content, self.c.d_rotary], dim=-1)  # content, rotary (not yet rotated)


# %% [latent] Latent c_KV
class Latent(nn.Module):
    """W↓KV compresses each token's 2,048 numbers to a 512-D latent c_KV and one 64-D key for RoPE.
    The normalized latent and the rotated key, 576 numbers, are all the KV cache keeps per token:
    multi-head attention with these heads would keep 16 × (192 + 128) = 5,120."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.down = nn.Linear(c.d_model, c.d_latent + c.d_rotary, bias=False)  # W↓KV: 2048 → 576
        self.norm = RMSNorm(c.d_latent, c.eps)

    def forward(self, x):
        latent, rotary_key = self.down(x).split([self.c.d_latent, self.c.d_rotary], dim=-1)
        return self.norm(latent), rotary_key[:, None]  # (batch, seq, 512), (batch, 1, seq, 64)


# %% [up] Up-projections W↑K · W↑V
class Up(nn.Module):
    """W↑K and W↑V expand the one cached latent into every head's own 128-D content key and value.
    (At inference W↑K can be folded into W_Q and W↑V into W_O, so the expansion never runs.)"""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        # W↑K and W↑V in one matrix, 512 → 16 heads × (128 key + 128 value)
        self.proj = nn.Linear(c.d_latent, c.n_heads * (c.d_content + c.d_value), bias=False)

    def forward(self, latent):  # latent: (batch, seq, 512)
        b, t, _ = latent.shape
        kv = self.proj(latent).view(b, t, self.c.n_heads, -1).transpose(1, 2)
        return kv.split([self.c.d_content, self.c.d_value], dim=-1)  # (batch, 16, seq, 128) each


# %% [rope] Decoupled RoPE
def yarn_frequencies(c: Config):
    """One frequency per pair of the 64 rotary dimensions, stretched by YaRN. Pairs that turn more
    than beta_fast times over the original 4,096 positions keep their frequency; pairs that turn
    less than beta_slow times are slowed by rope_factor; the pairs in between blend the two."""
    d = c.d_rotary
    frequencies = c.rope_theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)  # (d/2,)

    def pair(turns):  # the pair whose wavelength fits \`turns\` times into the original context
        return d * math.log(c.rope_original_positions / (turns * 2 * math.pi)) / (2 * math.log(c.rope_theta))

    low, high = max(math.floor(pair(c.rope_beta_fast)), 0), min(math.ceil(pair(c.rope_beta_slow)), d - 1)
    slowed = ((torch.arange(d // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    return frequencies / c.rope_factor * slowed + frequencies * (1 - slowed)


def rope(x, frequencies):
    """Rotate only the 64 rotary dimensions. As in DeepSeek's own code, the projection's
    interleaved pairs are first reordered to (evens, odds); then pair (i, i + 32) turns by
    position × frequency i. The content dimensions never rotate, so RoPE stays out of the latent."""
    x = x.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2)
    angles = torch.arange(x.shape[-2], dtype=torch.float32)[:, None] * frequencies  # (seq, 32)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


# %% [attention] Multi-head latent attention
def softmax_scale(c: Config):
    """1/√192 times YaRN's mscale², (0.1 × 0.707 × ln 40 + 1)²: 0.1147 for DeepSeek-V2-Lite."""
    mscale = 0.1 * c.rope_mscale * math.log(c.rope_factor) + 1
    return (c.d_content + c.d_rotary) ** -0.5 * mscale**2


def attention(q_content, q_rotary, k_content, k_rotary, v, scale):
    """Each head's score adds a content part, against its own key from the latent, and a rotary
    part, against the one rotated key every head shares (its head axis of size 1 broadcasts)."""
    t = q_content.shape[-2]
    scores = (q_content @ k_content.transpose(-2, -1) + q_rotary @ k_rotary.transpose(-2, -1)) * scale
    causal = torch.ones(t, t, dtype=torch.bool, device=scores.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v, weights  # each head's output, and the matrix the head card draws


# %% [output] Concatenate heads and project
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.n_heads * c.d_value, c.d_model, bias=False)  # W_O: 2048 → 2048

    def forward(self, heads):  # heads: (batch, 16, seq, 128)
        b, h, t, d = heads.shape
        return self.proj(heads.transpose(1, 2).reshape(b, t, h * d))


# %% [model] Multi-head latent attention layer
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.query = Query(c)
        self.latent = Latent(c)
        self.up = Up(c)
        self.output = Output(c)
        self.scale = softmax_scale(c)

    def forward(self, x):
        frequencies = yarn_frequencies(self.c)
        q_content, q_rotary = self.query(x)
        latent, k_rotary = self.latent(x)  # what the cache would store
        k_content, v = self.up(latent)
        heads, _ = attention(
            q_content, rope(q_rotary, frequencies), k_content, rope(k_rotary, frequencies), v, self.scale
        )
        return self.output(heads)
`;export{e as default};
