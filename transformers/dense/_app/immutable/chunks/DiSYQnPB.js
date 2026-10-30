const e=`# Multi-scale retention (RetNet; Sun et al., 2023), the Attention Explainer's RetNet tab. Retention
# replaces the softmax with a fixed causal decay: each head mixes its values with QKᵀ ⊙ D, and the
# same output comes from a recurrence over a d × d state. The tab is a formula demo on GPT-2 small's
# recorded queries, keys and values. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12  # GPT-2 small: 64-dimensional queries, keys and values per head
    value_dim: int = 768  # RetNet doubles it (value_factor 2); the tab uses GPT-2's 64-D values
    eps: float = 1e-6


# %% [rotation] Θ rotation of queries and keys
def rotation(t, head_dim):
    """xPos-style angles, one per pair of dimensions: position n turns pair i by n · θ_i."""
    theta = 1.0 / 10000 ** torch.linspace(0, 1, head_dim // 2)
    theta = theta.repeat_interleave(2)  # dimensions (0, 1), (2, 3), … share an angle
    phase = torch.arange(t)[:, None] * theta  # (seq, head_dim)
    return phase.sin(), phase.cos()


def rotate(x, sin, cos):
    """Q ⊙ Θ: rotate each pair (x₀, x₁) to (x₀ cos − x₁ sin, x₁ cos + x₀ sin)."""
    pairs = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    return x * cos + pairs * sin


# %% [decay] Decay mask D
def decays(n_heads):
    """γ_h = 1 − 2^(−5−h): every head forgets at its own fixed rate (multi-scale)."""
    return 1 - 2.0 ** (-5 - torch.arange(n_heads, dtype=torch.float))


def decay_mask(t, gamma):
    """D[h, n, m] = γ_h^(n−m) for m ≤ n, else 0 (eq. 5): causal, and fading with distance."""
    distance = torch.arange(t)[:, None] - torch.arange(t)  # (seq, seq)
    d = gamma[:, None, None] ** distance.clamp(min=0)
    return d * (distance >= 0)  # (heads, seq, seq)


# %% [parallel] Parallel retention
def parallel_retention(q, k, v, decay):
    """Retention(X) = (QKᵀ ⊙ D) V (eq. 5): no softmax, every pair weighted by the decay."""
    weights = (q @ k.transpose(-2, -1)) * decay  # (batch, heads, seq, seq)
    return weights @ v, weights


# %% [recurrent] Recurrent retention
def recurrent_retention(q, k, v, gamma):
    """S_n = γ S_{n−1} + k_nᵀ v_n, o_n = q_n S_n (eq. 4): the same output from a d × d state."""
    b, h, t, d = q.shape
    state = q.new_zeros(b, h, d, v.shape[-1])
    out = []
    for n in range(t):
        state = gamma[:, None, None] * state + k[..., n, :, None] * v[..., n, None, :]
        out.append(q[..., n, None, :] @ state)  # (batch, heads, 1, value_dim)
    return torch.cat(out, dim=-2)


# %% [output] GroupNorm, swish gate and W_O
class Output(nn.Module):
    """Each head's retention is RMS-normalized on its own (GroupNorm with one group per head, no
    affine), multiplied by swish(X W_G) and projected by W_O."""

    def __init__(self, c: Config):
        super().__init__()
        self.eps = c.eps
        self.g_proj = nn.Linear(c.d_model, c.value_dim, bias=False)
        self.out_proj = nn.Linear(c.value_dim, c.d_model, bias=False)

    def forward(self, heads, x):  # heads: (batch, heads, seq, head_dim); x: (batch, seq, d_model)
        b, h, t, d = heads.shape
        normed = heads * torch.rsqrt(heads.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = normed.transpose(1, 2).reshape(b, t, h * d)
        return self.out_proj(F.silu(self.g_proj(x)) * y)


# %% [model] Multi-scale retention layer
class MultiScaleRetention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.value_dim, bias=False)
        self.output = Output(c)
        self.register_buffer("gamma", decays(c.n_heads), persistent=False)

    def forward(self, x, recurrent=False):  # x: (batch, seq, d_model)
        b, t, _ = x.shape
        h = self.c.n_heads
        q = self.q_proj(x).view(b, t, h, -1).transpose(1, 2)
        k = self.k_proj(x).view(b, t, h, -1).transpose(1, 2) / math.sqrt(self.c.d_model // h)
        v = self.v_proj(x).view(b, t, h, -1).transpose(1, 2)
        sin, cos = rotation(t, q.shape[-1])
        q, k = rotate(q, sin, cos), rotate(k, sin, cos)
        if recurrent:
            heads = recurrent_retention(q, k, v, self.gamma)
        else:
            heads, _ = parallel_retention(q, k, v, decay_mask(t, self.gamma))
        return self.output(heads, x)
`;export{e as default};
