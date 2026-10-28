const e=`# Gated attention ("Gated Attention for Large Language Models", Qiu et al., 2025) as Qwen3-Next
# ships it, the model in the Attention Explainer's Gated tab: grouped-query attention whose output is
# multiplied, head by head and dimension by dimension, by a sigmoid gate computed from the same
# input. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 2  # 8 query heads share each key-value head
    head_dim: int = 256
    rotary_dim: int = 64  # RoPE on the first quarter of each head
    rope_theta: float = 10_000_000.0
    eps: float = 1e-6


# %% [projections] Query, gate, key and value projections
class Projections(nn.Module):
    """q_proj gives every head a query and a gate, side by side; k_proj and v_proj give the
    key-value heads."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.d_model, c.n_heads * c.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)

    def forward(self, x):  # x: (batch, seq, d_model), after the input RMSNorm
        b, t, _ = x.shape
        q, gate = self.q_proj(x).view(b, t, self.c.n_heads, 2 * self.c.head_dim).chunk(2, dim=-1)
        k = self.k_proj(x).view(b, t, self.c.n_kv_heads, self.c.head_dim)
        v = self.v_proj(x).view(b, t, self.c.n_kv_heads, self.c.head_dim)
        return q, gate, k, v  # (batch, seq, heads, head_dim) each


# %% [qk-norm] RMSNorm on each query and key head
class HeadRMSNorm(nn.Module):
    """Zero-centered gain: the stored weight starts at 0 and the scale is 1 + weight."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight)


# %% [rope] Rotary embedding on the first 64 of 256 dimensions
def rope(x, positions, c: Config):
    """x: (batch, heads, seq, head_dim). Rotates pairs (i, i + 32) of the first 64 dims."""
    steps = torch.arange(0, c.rotary_dim, 2, dtype=torch.float32, device=x.device) / c.rotary_dim
    angles = positions[:, None].float() * (1.0 / c.rope_theta**steps)[None]  # (seq, 32)
    cos, sin = torch.cat([angles, angles], dim=-1).cos(), torch.cat([angles, angles], dim=-1).sin()
    turned, rest = x[..., : c.rotary_dim], x[..., c.rotary_dim :]
    half = c.rotary_dim // 2
    swapped = torch.cat([-turned[..., half:], turned[..., :half]], dim=-1)
    return torch.cat([turned * cos + swapped * sin, rest], dim=-1)


# %% [attention] Grouped-query causal attention
def attention(q, k, v):
    """q: (batch, heads, seq, d); k, v: (batch, kv_heads, seq, d), shared by groups of heads."""
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


# %% [gate] Sigmoid output gate
def output_gate(heads, gate):
    """heads, gate: (batch, seq, heads, d). Each head dimension is scaled by its own score in
    (0, 1); in Qwen3-Next most scores are far below 0.1."""
    return heads * torch.sigmoid(gate)


# %% [output] Output projection
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.o_proj = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def forward(self, gated):  # gated: (batch, seq, heads, head_dim)
        return self.o_proj(gated.flatten(-2))


# %% [model] Gated attention layer
class GatedAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.proj = Projections(c)
        self.q_norm = HeadRMSNorm(c.head_dim, c.eps)
        self.k_norm = HeadRMSNorm(c.head_dim, c.eps)
        self.output = Output(c)

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device)
        q, gate, k, v = self.proj(x)
        q = rope(self.q_norm(q).transpose(1, 2), positions, self.c)
        k = rope(self.k_norm(k).transpose(1, 2), positions, self.c)
        heads, _ = attention(q, k, v.transpose(1, 2))
        return self.output(output_gate(heads.transpose(1, 2), gate))
`;export{e as default};
