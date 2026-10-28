const e=`# Multi-head attention as in GPT-2 small, the model in the Attention Explainer's MHA tab.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12  # 768 / 12 = 64 dimensions per head


# %% [qkv] Q · K · V projection
class QKV(nn.Module):
    """One linear map gives every head its query, key and value."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.proj = nn.Linear(c.d_model, 3 * c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model), the input after LayerNorm
        b, t, d = x.shape
        q, k, v = self.proj(x).split(d, dim=-1)
        # Split into heads: (batch, heads, seq, head_dim)
        return tuple(z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in (q, k, v))


# %% [attention] Causal scaled dot-product attention
def attention(q, k, v):
    """Each query takes a softmax over the scaled scores of itself and earlier keys."""
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (batch, heads, seq, seq)
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v, weights  # each head's output, and the matrix the head card draws


# %% [output] Concatenate heads and project
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.d_model, c.d_model)  # W_O

    def forward(self, heads):  # heads: (batch, heads, seq, head_dim)
        b, h, t, d = heads.shape
        return self.proj(heads.transpose(1, 2).reshape(b, t, h * d))


# %% [model] Multi-head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.qkv = QKV(c)
        self.output = Output(c)

    def forward(self, x):
        q, k, v = self.qkv(x)
        heads, _ = attention(q, k, v)
        return self.output(heads)
`;export{e as default};
