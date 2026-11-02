const e=`# YaRN as Qwen documents it for Qwen2.5 (rope_scaling type "yarn", factor 4, original context 32,768),
# applied to Qwen2.5-0.5B in the Attention Explainer's YaRN tab. Each "# %% [id]" section is one step.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 896
    n_heads: int = 14  # query heads, 64 dimensions each
    n_kv_heads: int = 2  # key/value heads; each is shared by 7 query heads
    rope_theta: float = 1_000_000.0
    factor: float = 4.0  # stretch the 32,768 trained positions to 4 × 32,768 = 131,072
    original_context: int = 32_768  # the context RoPE was trained on
    beta_fast: float = 32  # pairs that turn more than 32 times in the original context keep θ_i
    beta_slow: float = 1  # pairs that turn less than once are divided by the factor


# %% [qkv] Q · K · V projection
class QKV(nn.Module):
    """Three projections with a bias, unchanged by YaRN."""

    def __init__(self, c: Config):
        super().__init__()
        self.head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * self.head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, _ = x.shape

        def heads(z):  # (batch, seq, n · 64) → (batch, n, seq, 64)
            return z.view(b, t, -1, self.head_dim).transpose(1, 2)

        return heads(self.q(x)), heads(self.k(x)), heads(self.v(x))


# %% [frequencies] Interpolate, extrapolate or ramp each pair
def frequencies(c: Config, head_dim):
    """YaRN ("NTK-by-parts"): pair i turns L/λ_i times in the original context L. Fast pairs (more than
    β_fast turns) keep θ_i; slow pairs (fewer than β_slow) get θ_i / s, so position m·s turns them as
    far as m did in training; the pairs between are blended by a linear ramp γ_i."""
    wavelength = c.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)  # 1 / θ_i

    def pair(turns):  # the pair index that turns \`turns\` times in the original context
        return head_dim * math.log(c.original_context / (turns * 2 * math.pi)) / (2 * math.log(c.rope_theta))

    low = max(math.floor(pair(c.beta_fast)), 0)
    high = min(math.ceil(pair(c.beta_slow)), head_dim - 1)
    ramp = ((torch.arange(head_dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)  # γ_i
    keep = 1 - ramp  # the share of θ_i a pair keeps
    return 1.0 / (c.factor * wavelength) * (1 - keep) + 1.0 / wavelength * keep  # (32,)


# %% [temperature] Attention temperature
def temperature(c: Config):
    """√(1/t) = 0.1 · ln s + 1. It multiplies cos and sin, so every score q·k/√d is multiplied by its
    square: a sharper softmax over the longer context."""
    return 0.1 * math.log(c.factor) + 1.0 if c.factor > 1 else 1.0


# %% [rope] Rotate with the stretched frequencies
def rope(x, positions, c: Config):
    """RoPE with YaRN's θ_i, and cos and sin scaled by the temperature factor."""
    d = x.shape[-1]
    angles = positions.float()[:, None] * frequencies(c, d)  # (seq, 32)
    scale = temperature(c)
    cos, sin = (angles.cos() * scale).repeat(1, 2), (angles.sin() * scale).repeat(1, 2)
    turned = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)
    return x * cos + turned * sin


# %% [attention] Attention on the rotated queries and keys
def attention(q, k, v):
    """Grouped-query causal attention; the √d scale is the model's, the temperature is already in q and k."""
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


# %% [output] Concatenate heads and project
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, heads):  # heads: (batch, 14, seq, 64)
        b, h, t, d = heads.shape
        return self.proj(heads.transpose(1, 2).reshape(b, t, h * d))


# %% [model] Attention layer with YaRN
class YarnAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.qkv = QKV(c)
        self.output = Output(c)

    def forward(self, x, positions=None):
        q, k, v = self.qkv(x)
        positions = torch.arange(x.shape[1], device=x.device) if positions is None else positions
        heads, _ = attention(rope(q, positions, self.c), rope(k, positions, self.c), v)
        return self.output(heads)
`;export{e as default};
