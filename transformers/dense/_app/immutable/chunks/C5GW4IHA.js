const e=`# ALiBi attention as in BLOOM-560m, the model in the Attention Explainer's ALiBi tab.
# BLOOM has no position embeddings: each head adds a linear distance penalty to its scores.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    vocab_size: int = 250880
    d_model: int = 1024
    n_heads: int = 16  # 1024 / 16 = 64 dimensions per head
    eps: float = 1e-5


# %% [embedding] Token embedding, no position
class Embedding(nn.Module):
    """A token lookup and a LayerNorm. Nothing depends on the position, so the same token enters block 1
    with the same vector wherever it stands; order reaches the model only through the ALiBi bias."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, ids):  # ids: (batch, seq)
        return self.norm(self.token(ids))  # (batch, seq, d_model)


# %% [qkv] Q · K · V projection
class QKV(nn.Module):
    """One linear map; BLOOM lays its output out as (heads, 3, head_dim), so each head's q, k and v sit together."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.proj = nn.Linear(c.d_model, 3 * c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model), the input after input_layernorm
        b, t, d = x.shape
        fused = self.proj(x).view(b, t, self.n_heads, 3, d // self.n_heads)
        return tuple(fused[..., part, :].transpose(1, 2) for part in range(3))  # each (batch, heads, seq, head_dim)


# %% [slopes] ALiBi slopes
def alibi_slopes(n_heads):
    """A geometric sequence, one slope per head: 2^(−8h/n) for h = 1 … n when n is a power of two
    (BLOOM-560m: 2^(−h/2), from 0.71 down to 0.0039). Otherwise the closest power of two's slopes,
    then every other slope of the next power of two."""
    closest = 2 ** math.floor(math.log2(n_heads))
    slopes = [2 ** (-8 * h / closest) for h in range(1, closest + 1)]
    slopes += [2 ** (-8 * h / (2 * closest)) for h in range(1, 2 * (n_heads - closest), 2)]
    return torch.tensor(slopes)


# %% [bias] Distance bias m · (j − i)
def alibi_bias(slopes, t):
    """(heads, seq, seq): key j's score for query i drops by m · (i − j), linearly with the distance.
    Steep heads attend locally; the flattest ones can still reach far back."""
    distance = torch.arange(t)[None, :] - torch.arange(t)[:, None]  # j − i, ≤ 0 for visible keys
    return slopes[:, None, None] * distance


# %% [attention] Attention with the ALiBi bias
def attention(q, k, v, bias):
    """Causal softmax over the scaled scores plus each head's distance bias."""
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) + bias  # (batch, heads, seq, seq)
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


# %% [model] BLOOM attention layer
class ALiBiAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.qkv = QKV(c)
        self.output = Output(c)
        self.register_buffer("slopes", alibi_slopes(c.n_heads), persistent=False)

    def forward(self, x):
        q, k, v = self.qkv(x)
        heads, _ = attention(q, k, v, alibi_bias(self.slopes, x.shape[1]).to(x.device))
        return self.output(heads)
`;export{e as default};
