const e=`# MHA, MQA and GQA as one function of the number of key/value heads, for the Attention Explainer's
# KV heads Compare tab: GPT-2 small (12 key/value heads for 12 query heads), SantaCoder (1 for 16)
# and Qwen2.5-0.5B (2 for 14). Each "# %% [id]" section is one step of the comparison.
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    layers: int = 24
    n_heads: int = 14  # query heads
    n_kv_heads: int = 2  # key/value heads: n_heads in MHA, 1 in MQA, a divisor in between in GQA
    head_dim: int = 64


# %% [attention] Query heads sharing key/value heads
def attention(q, k, v):
    """q: (batch, n_heads, seq, d); k, v: (batch, n_kv_heads, seq, d). Query head h reads key/value
    head h // group, group = n_heads / n_kv_heads: 1 in MHA, all of them in MQA."""
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v, weights  # each head's output, and the map the Compare tab draws


# %% [decode] One decoding step from the cache
def decode(q, k_cache, v_cache):
    """q: (batch, n_heads, 1, d), the new token's queries; the cache holds (batch, n_kv_heads, seq, d).
    A group's query heads become the rows of one product, so each cached key and value is read once,
    not once per query head: a smaller cache is less memory to read at every step."""
    b, h, _, d = q.shape
    g = k_cache.shape[1]
    rows = q.reshape(b, g, h // g, d)  # (batch, n_kv_heads, group, d): heads 0..group-1 are group 0's rows
    weights = (rows @ k_cache.transpose(-2, -1) / math.sqrt(d)).softmax(dim=-1)
    return (weights @ v_cache).reshape(b, h, 1, d)


# %% [cache] KV-cache size
def cache_bytes(c: Config, tokens: int, bytes_per_number: int = 2):
    """Every layer keeps one key and one value of head_dim numbers per key/value head per token:
    GPT-2 small 2 × 12 × 12 × 64 = 18,432 numbers a token, SantaCoder and Qwen2.5-0.5B 6,144."""
    return 2 * c.layers * c.n_kv_heads * c.head_dim * tokens * bytes_per_number


# %% [pool] Converting MHA by mean-pooling
def pool(weight, bias, n_kv_heads, head_dim):
    """A multi-head key (or value) projection, weight (n_heads · d, d_model) and bias (n_heads · d,),
    turned into n_kv_heads heads: each group of consecutive heads is replaced by its mean (Ainslie et
    al., 2023). The Compare tab's GPT-2 runs start from this and then train further."""
    d_model = weight.shape[1]
    w = weight.view(n_kv_heads, -1, head_dim, d_model).mean(dim=1).reshape(-1, d_model)
    return w, bias.view(n_kv_heads, -1, head_dim).mean(dim=1).reshape(-1)
`;export{e as default};
