const e=`# Grouped-query attention as in Qwen2.5-0.5B, the model in the Attention Explainer's GQA tab.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 896
    n_heads: int = 14  # query heads, 896 / 14 = 64 dimensions each
    n_kv_heads: int = 2  # key/value heads; each is shared by 14 / 2 = 7 query heads
    rope_theta: float = 1_000_000.0


# %% [qkv] Q · K · V projection
class QKV(nn.Module):
    """Three projections with a bias: 14 query heads, but only 2 key heads and 2 value heads."""

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


# %% [rope] RoPE
def rope(x, theta):
    """Rotate each pair (i, i + 32) of a query or key by its position × θ^(−2i/64). Values never
    rotate. The two key heads are rotated once, before any query head reads them."""
    d = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(x.shape[-2], dtype=torch.float32, device=x.device)[:, None] * frequencies
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)  # (seq, 64)
    turned = torch.cat((-x[..., d // 2 :], x[..., : d // 2]), dim=-1)
    return x * cos + turned * sin


# %% [attention] Grouped-query attention
def attention(q, k, v):
    """Query head h reads key/value head h // 7: heads 1–7 share the first, 8–14 the second."""
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


# %% [model] Grouped-query attention layer
class GroupedQueryAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.theta = c.rope_theta
        self.qkv = QKV(c)
        self.output = Output(c)

    def forward(self, x):
        q, k, v = self.qkv(x)
        heads, _ = attention(rope(q, self.theta), rope(k, self.theta), v)
        return self.output(heads)
`;export{e as default};
