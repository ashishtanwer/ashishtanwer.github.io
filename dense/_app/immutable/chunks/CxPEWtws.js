const e=`# Multi-query attention as in SantaCoder (GPTBigCode with multi_query=True), the model in the
# Attention Explainer's MQA tab. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 2048
    n_heads: int = 16  # 2048 / 16 = 128 dimensions per head; there is one key head and one value head


# %% [qkv] Q · K · V projection
class QKV(nn.Module):
    """One linear map gives every token 16 query heads but a single key and a single value."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.head_dim = c.d_model // c.n_heads
        self.proj = nn.Linear(c.d_model, c.d_model + 2 * self.head_dim)  # 2048 → 2048 + 128 + 128

    def forward(self, x):  # x: (batch, seq, d_model), the input after LayerNorm
        b, t, d = x.shape
        q, k, v = self.proj(x).split([d, self.head_dim, self.head_dim], dim=-1)
        q = q.view(b, t, self.n_heads, -1).transpose(1, 2)  # (batch, 16, seq, 128)
        # One key head and one value head: (batch, 1, seq, 128). The KV cache keeps only these,
        # 256 numbers per token per layer instead of multi-head attention's 4,096.
        return q, k[:, None], v[:, None]


# %% [attention] Multi-query attention
def attention(q, k, v):
    """Every query head scores the same keys and mixes the same values: the size-1 head axis of
    k and v broadcasts across the 16 query heads, so nothing is copied."""
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (batch, 16, seq, seq)
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v, weights  # each head's output, and the matrix the head card draws


# %% [output] Concatenate heads and project
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.d_model, c.d_model)  # W_O

    def forward(self, heads):  # heads: (batch, 16, seq, 128)
        b, h, t, d = heads.shape
        return self.proj(heads.transpose(1, 2).reshape(b, t, h * d))


# %% [model] Multi-query attention layer
class MultiQueryAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.qkv = QKV(c)
        self.output = Output(c)

    def forward(self, x):
        q, k, v = self.qkv(x)
        heads, _ = attention(q, k, v)
        return self.output(heads)
`;export{e as default};
