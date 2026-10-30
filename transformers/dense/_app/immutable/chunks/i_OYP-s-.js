const e=`# DeepSeek Sparse Attention (DSA; DeepSeek-V3.2-Exp, 2025), the Attention Explainer's DSA tab. A
# lightning indexer scores every earlier token for each query, the query keeps its top-k tokens,
# and attention runs over those tokens only. The tab is a formula demo on GPT-2 small's recorded
# queries, keys and values. GPT-2 has no indexer, so the tab ranks tokens with the stand-in below.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12  # GPT-2 small: 64-dimensional queries, keys and values per head
    index_heads: int = 64  # H^I, as in V3.2
    index_dim: int = 128  # d^I, as in V3.2
    top_k: int = 8  # tokens each query keeps (V3.2: 2048 of up to 128K)


def causal(t):
    return torch.ones(t, t, dtype=torch.bool).tril()


# %% [indexer] Lightning indexer
class LightningIndexer(nn.Module):
    """I[t, s] = Σ_j w[t, j] · ReLU(q[t, j] · k[s]) (eq. 1): a few small heads that share one key
    per token, with a ReLU instead of a softmax. V3.2 also puts RoPE on part of q and k, rotates
    both with a Hadamard transform and runs in FP8; those are left out here."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads, self.dim = c.index_heads, c.index_dim
        self.wq = nn.Linear(c.d_model, c.index_heads * c.index_dim, bias=False)
        self.wk = nn.Linear(c.d_model, c.index_dim, bias=False)
        self.k_norm = nn.LayerNorm(c.index_dim, eps=1e-6)
        self.weights = nn.Linear(c.d_model, c.index_heads, bias=False)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.heads, self.dim)
        k = self.k_norm(self.wk(x))  # (batch, seq, index_dim): one key per token
        w = self.weights(x) * self.heads**-0.5 * self.dim**-0.5  # (batch, seq, index_heads)
        dots = torch.einsum("bthd,bsd->bths", q, k).relu()
        return torch.einsum("bth,bths->bts", w, dots)  # (batch, query, key)


# %% [stand-in] Stand-in indexer: the indexer's training target
def oracle_index(q, k):
    """The tab's stand-in. DSA trains its indexer to match the layer's dense attention, summed over
    heads and L1-normalized (eq. 3), so this ranks keys by that target: what a perfectly trained
    indexer would score. Computing it needs the dense attention DSA avoids, so it is a demo only."""
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    dense = scores.masked_fill(~causal(t), float("-inf")).softmax(dim=-1)  # (batch, heads, seq, seq)
    p = dense.sum(dim=1)  # (batch, seq, seq): one score per query and key, shared by every head
    return p / p.sum(dim=-1, keepdim=True)


# %% [topk] Top-k token selection
def select_tokens(index, top_k):
    """Each query keeps the top_k earlier tokens by index score, or all of them while it has fewer
    than top_k. One selection per query, shared by every head."""
    t = index.shape[-1]
    top = index.masked_fill(~causal(t), float("-inf")).topk(min(top_k, t), dim=-1).indices
    return torch.zeros_like(index, dtype=torch.bool).scatter(-1, top, True) & causal(t)


# %% [sparse-attention] Attention over the selected tokens
def sparse_attention(q, k, v, selected):
    """u_t = Attn(h_t, {c_s : s selected}) (eq. 2): the softmax runs over the selected tokens only."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = scores.masked_fill(~selected.unsqueeze(1), float("-inf")).softmax(dim=-1)
    return weights @ v, weights


# %% [model] DSA layer
class SparseAttention(nn.Module):
    """Q, K, V and the indexer from the same input, top-k, sparse attention, then W_O. V3.2 builds
    this on MLA in its MQA mode; here the heads keep their own keys and values, as in GPT-2."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.indexer = LightningIndexer(c)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (z.view(b, t, self.c.n_heads, -1).transpose(1, 2) for z in self.qkv(x).split(d, dim=-1))
        selected = select_tokens(self.indexer(x), self.c.top_k)
        heads, _ = sparse_attention(q, k, v, selected)
        return self.out(heads.transpose(1, 2).reshape(b, t, d))
`;export{e as default};
