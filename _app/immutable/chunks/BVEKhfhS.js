const e=`# Sliding-window attention as in Gemma 3 270M, the model in the Attention Explainer's Window tab.
# Five local blocks (window 512) alternate with one global block. Each "# %% [id]" section is one
# step of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 640
    n_heads: int = 4
    n_kv_heads: int = 1  # the four query heads share one key/value head
    head_dim: int = 256
    window: int = 512  # a local query sees the last 512 keys, itself included
    rope_local_theta: float = 10_000.0
    rope_global_theta: float = 1_000_000.0
    query_pre_attn_scalar: float = 256.0  # scores are scaled by 1/√256
    eps: float = 1e-6


class RMSNorm(nn.Module):
    """Gemma's RMSNorm scales by (1 + γ)."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        return (1 + self.weight) * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) by position × theta^(-2i/d)."""
    d = x.shape[-1]
    angles = positions[:, None].float() * theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [qkv] Q · K · V with QK-norm and RoPE
class QKV(nn.Module):
    """Project, normalize each query and key head, then rotate them by position. Local blocks rotate
    with θ = 10,000 and global blocks with θ = 1,000,000; the mask, not RoPE, limits a local block."""

    def __init__(self, c: Config, local: bool):
        super().__init__()
        self.c, self.theta = c, c.rope_local_theta if local else c.rope_global_theta
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.eps)
        self.k_norm = RMSNorm(c.head_dim, c.eps)

    def forward(self, x):  # x: (batch, seq, d_model), the input after input_layernorm
        b, t, _ = x.shape
        c, positions = self.c, torch.arange(t, device=x.device)
        q = self.q(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = self.k(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        return rope(self.q_norm(q), positions, self.theta), rope(self.k_norm(k), positions, self.theta), v


# %% [mask] Sliding-window mask
def window_mask(t, window=None):
    """(seq, seq) booleans: query i sees key j when j ≤ i and, in a local block, i − j < window.
    A global block passes window=None and keeps the plain causal mask."""
    distance = torch.arange(t)[:, None] - torch.arange(t)[None, :]  # i − j
    visible = distance >= 0
    return visible & (distance < window) if window is not None else visible


# %% [attention] Local and global attention
def attention(q, k, v, mask, scale):
    """Softmax over the visible keys only; the one key/value head is shared by every query head."""
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    scores = q @ k.transpose(-2, -1) * scale  # (batch, heads, seq, seq)
    weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return weights @ v, weights  # each head's output, and the matrix the head card draws


# %% [output] Concatenate heads and project
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)  # W_O

    def forward(self, heads):  # heads: (batch, heads, seq, head_dim)
        b, h, t, d = heads.shape
        return self.proj(heads.transpose(1, 2).reshape(b, t, h * d))


# %% [model] Gemma 3 attention layer, local or global
class SlidingWindowAttention(nn.Module):
    def __init__(self, c: Config = Config(), local: bool = True):
        super().__init__()
        self.c, self.local = c, local
        self.qkv = QKV(c, local)
        self.output = Output(c)

    def forward(self, x):
        q, k, v = self.qkv(x)
        mask = window_mask(x.shape[1], self.c.window if self.local else None).to(x.device)
        heads, _ = attention(q, k, v, mask, self.c.query_pre_attn_scalar**-0.5)
        return self.output(heads)

`;export{e as default};
