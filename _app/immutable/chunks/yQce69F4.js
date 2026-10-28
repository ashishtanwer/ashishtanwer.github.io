const e=`# Attention sinks and StreamingLLM in GPT-2 small, the model in the Attention Explainer's Sinks tab.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12
    sinks: int = 4  # StreamingLLM keeps the first 4 tokens ...
    cache_size: int = 64  # ... and the most recent ones, 64 keys per query in all


# %% [attention] Causal attention with a KV cache
class Attention(nn.Module):
    """GPT-2's attention. Without a cache it reads the whole causal context; with one it decodes a token
    at a time, its key and value joining the cache and its query reading every cached entry."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)  # W_O

    def forward(self, x, cache=None):  # x: (batch, seq, d_model), the input after LayerNorm
        b, t, d = x.shape
        q, k, v = (z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in self.qkv(x).split(d, dim=-1))
        if cache is not None:
            k, v = cache.update(k, v)  # the cached keys and values, then this token's
        n = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (batch, heads, seq, keys)
        causal = torch.ones(t, n, dtype=torch.bool, device=x.device).tril(n - t)
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        heads = weights @ v
        return self.proj(heads.transpose(1, 2).reshape(b, t, d)), weights


# %% [sink-mass] Attention on the first token
def sink_mass(weights):
    """Mean weight on key 0 over every query but the first (which can only see key 0): (..., heads).
    Most of GPT-2's later heads put a third to nine tenths of their attention there, whatever the token."""
    return weights[..., 1:, 0].mean(dim=-1)


# %% [cache] StreamingLLM KV cache
class SinkCache:
    """Keeps the first \`sinks\` entries and the most recent ones: between steps at most \`size − 1\`, so with
    its own key a query sees \`size\`. sinks=0 is a plain window of the last \`size\` tokens. Each cached key
    keeps the position it was computed at (GPT-2 adds positions to the input, before the cache)."""

    def __init__(self, size: int, sinks: int):
        self.size, self.sinks = size, sinks
        self.keys = self.values = None  # (batch, heads, cached, head_dim)

    def update(self, k, v):
        self.keys = k if self.keys is None else torch.cat([self.keys, k], dim=2)
        self.values = v if self.values is None else torch.cat([self.values, v], dim=2)
        keys, values = self.keys, self.values
        excess = self.keys.shape[2] - (self.size - 1)
        if excess > 0:  # evict the oldest entries after the sinks
            s = self.sinks
            self.keys = torch.cat([self.keys[:, :, :s], self.keys[:, :, s + excess :]], dim=2)
            self.values = torch.cat([self.values[:, :, :s], self.values[:, :, s + excess :]], dim=2)
        return keys, values


# %% [loss] Next-token loss
def next_token_loss(logits, ids):
    """−log p(next token) at every position but the last: (batch, seq − 1), in nats."""
    return F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
`;export{e as default};
