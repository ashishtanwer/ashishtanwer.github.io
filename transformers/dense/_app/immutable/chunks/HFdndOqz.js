const e=`# Rotary position embedding (RoPE) as in Qwen2.5-0.5B, the model in the Attention Explainer's RoPE tab.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 896
    n_heads: int = 14  # query heads, 896 / 14 = 64 dimensions each
    n_kv_heads: int = 2  # key/value heads; each is shared by 7 query heads
    rope_theta: float = 1_000_000.0  # the base; trained up to 32,768 positions


# %% [qkv] Q · K · V projection
class QKV(nn.Module):
    """Three projections with a bias. Nothing here knows where a token is."""

    def __init__(self, c: Config):
        super().__init__()
        self.head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * self.head_dim)  # 896 → 896
        self.k = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim)  # 896 → 128
        self.v = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim)  # 896 → 128

    def forward(self, x):  # x: (batch, seq, d_model), the input after RMSNorm
        b, t, _ = x.shape

        def heads(z):  # (batch, seq, n · 64) → (batch, n, seq, 64)
            return z.view(b, t, -1, self.head_dim).transpose(1, 2)

        return heads(self.q(x)), heads(self.k(x)), heads(self.v(x))


# %% [frequencies] One frequency per pair
def frequencies(head_dim, theta):
    """θ_i = base^(−2i/64) for the 32 pairs: pair 0 turns 1 radian per token, pair 31 about 1.5e-6.
    The wavelength 2π/θ_i, in tokens, runs from 6.3 to about 4 million."""
    return 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)  # (32,)


# %% [rope] Rotate each pair by position × θ_i
def rope(x, positions, theta):
    """Pair i is the coordinates (i, i + 32). A token at position m turns every pair by the angle m·θ_i;
    lengths do not change. Queries and keys are rotated, values never."""
    d = x.shape[-1]
    angles = positions.float()[:, None] * frequencies(d, theta)  # (seq, 32)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)  # (seq, 64): both halves alike
    turned = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)  # (−x_b, x_a)
    return x * cos + turned * sin


# %% [relative] The score sees only i − j
def relative_score(q, k, distance, theta):
    """(R_i q) · (R_j k) = q · R_(j−i) k: the rotated score of an unrotated query q and key k, from their
    distance i − j alone. Pair by pair: (q_a k_a + q_b k_b) cos(Δθ) + (q_a k_b − q_b k_a) sin(Δθ)."""
    d = q.shape[-1]
    qa, qb, ka, kb = q[..., : d // 2], q[..., d // 2 :], k[..., : d // 2], k[..., d // 2 :]
    angle = distance[..., None] * frequencies(d, theta)
    return ((qa * ka + qb * kb) * angle.cos() + (qa * kb - qb * ka) * angle.sin()).sum(-1)


# %% [attention] Attention on rotated queries and keys
def attention(q, k, v):
    """Grouped-query attention: query head h reads key/value head h // 7."""
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)  # (batch, 14, seq, 64)
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v, weights  # each head's output, and the matrix the head card draws


# %% [output] Concatenate heads and project
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)  # W_O, no bias

    def forward(self, heads):  # heads: (batch, 14, seq, 64)
        b, h, t, d = heads.shape
        return self.proj(heads.transpose(1, 2).reshape(b, t, h * d))


# %% [model] Attention layer with RoPE
class RotaryAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.theta = c.rope_theta
        self.qkv = QKV(c)
        self.output = Output(c)

    def forward(self, x, positions=None):
        q, k, v = self.qkv(x)
        positions = torch.arange(x.shape[1], device=x.device) if positions is None else positions
        heads, _ = attention(rope(q, positions, self.theta), rope(k, positions, self.theta), v)
        return self.output(heads)
`;export{e as default};
