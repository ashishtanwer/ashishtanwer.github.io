const e=`# Native Sparse Attention (NSA; Yuan et al., 2025), the Attention Explainer's NSA tab. The tab is a
# formula demo: it runs these functions on GPT-2 small's recorded queries, keys and values. Each
# query mixes three branches: attention over compressed blocks, attention over the few blocks
# those scores rank highest, and a sliding window. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12  # GPT-2 small: 64-dimensional queries, keys and values per head
    block: int = 4  # l = d = l': one blocking for compression and selection (paper: l=32, d=16, l'=64)
    top_n: int = 3  # blocks each query selects, its own block included (paper: 16)
    window: int = 8  # sliding-window length w (paper: 512)


def causal(t):
    return torch.ones(t, t, dtype=torch.bool).tril()


# %% [compress] Compressed keys and values
def compress(k, v, block):
    """φ turns each block of \`block\` tokens into one key and one value (eq. 7). NSA learns φ, an
    MLP with intra-block positions; this demo, like fla's NSA kernels, takes the block mean."""
    t = k.shape[-2] // block * block  # a trailing partial block is not compressed yet

    def pool(z):  # (batch, heads, seq, dim) → (batch, heads, blocks, dim)
        return z[..., :t, :].unflatten(-2, (-1, block)).mean(dim=-2)

    return pool(k), pool(v)


# %% [compressed] Attention over the compressed blocks
def compressed_attention(q, k_cmp, v_cmp, block):
    """Query t sees block i once all its tokens are in, (i + 1) · block − 1 ≤ t (eq. 8)."""
    t, n = q.shape[-2], k_cmp.shape[-2]
    visible = (torch.arange(n) + 1) * block - 1 <= torch.arange(t)[:, None]  # (seq, blocks)
    scores = q @ k_cmp.transpose(-2, -1) / math.sqrt(q.shape[-1])
    p = scores.masked_fill(~visible, float("-inf")).softmax(dim=-1)
    p = p.nan_to_num(0.0)  # queries before the first full block get nothing from this branch
    return p @ v_cmp, p  # the branch's output, and p_cmp, which also ranks the blocks


# %% [select] Top-n block selection
def select_blocks(p_cmp, t, block, top_n):
    """Selection blocks match the compression blocks, so a block's importance is p_cmp itself
    (eq. 9). The query's own block, which compression cannot see yet, is always kept, as in fla's
    NSA; the other n − 1 are the earlier blocks with the highest importance (eq. 11)."""
    n_blocks = -(-t // block)
    own = torch.arange(t)[:, None] // block  # (seq, 1)
    j = torch.arange(n_blocks)  # (blocks,)
    importance = torch.zeros(*p_cmp.shape[:-1], n_blocks, dtype=p_cmp.dtype)
    importance[..., : p_cmp.shape[-1]] = p_cmp
    importance = importance.masked_fill(j == own, float("inf")).masked_fill(j > own, float("-inf"))
    top = importance.topk(min(top_n, n_blocks), dim=-1).indices
    return torch.zeros_like(importance, dtype=torch.bool).scatter(-1, top, True) & (j <= own)


# %% [selected] Attention over the selected blocks
def selected_attention(q, k, v, chosen, block):
    """Causal attention over the tokens of the chosen blocks only (eq. 12)."""
    t = q.shape[-2]
    keep = chosen.repeat_interleave(block, dim=-1)[..., :t] & causal(t)  # (batch, heads, seq, seq)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = scores.masked_fill(~keep, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


# %% [window] Sliding-window attention
def window_attention(q, k, v, window):
    """Causal attention over the last \`window\` tokens, the query's included."""
    t = q.shape[-2]
    i, j = torch.arange(t)[:, None], torch.arange(t)
    keep = (j <= i) & (j > i - window)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = scores.masked_fill(~keep, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


# %% [gate] Gated sum of the three branches
def gated_sum(branches, gates):
    """o_t = g_cmp · o_cmp + g_slc · o_slc + g_win · o_win (eq. 5); each gate is per token and head."""
    return sum(g * o for g, o in zip(gates, branches, strict=True))


def mixing_matrix(p_cmp, w_slc, w_win, gates, block):
    """The same sum as one token-mixing matrix: a block mean spreads p_cmp evenly over the block's
    tokens, so mixing_matrix(...) @ v equals the gated sum of the branch outputs."""
    w_cmp = torch.zeros_like(w_slc)
    w_cmp[..., : p_cmp.shape[-1] * block] = p_cmp.repeat_interleave(block, dim=-1) / block
    return gated_sum((w_cmp, w_slc, w_win), gates)


# %% [model] NSA layer
class NativeSparseAttention(nn.Module):
    """One NSA layer: Q, K, V, the three branches, sigmoid gates from the input, then W_O. The tab
    replaces the learned gates with equal gates g = 1/3 and draws GPT-2's own W_O."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.gate = nn.Linear(c.d_model, 3 * c.n_heads)  # g_cmp, g_slc, g_win for every head
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        q, k, v = (z.view(b, t, self.c.n_heads, -1).transpose(1, 2) for z in self.qkv(x).split(d, dim=-1))
        k_cmp, v_cmp = compress(k, v, self.c.block)
        o_cmp, p_cmp = compressed_attention(q, k_cmp, v_cmp, self.c.block)
        chosen = select_blocks(p_cmp, t, self.c.block, self.c.top_n)
        o_slc, _ = selected_attention(q, k, v, chosen, self.c.block)
        o_win, _ = window_attention(q, k, v, self.c.window)
        # (3, batch, heads, seq, 1): one gate per branch, token and head
        gates = torch.sigmoid(self.gate(x)).view(b, t, self.c.n_heads, 3).permute(3, 0, 2, 1).unsqueeze(-1)
        heads = gated_sum((o_cmp, o_slc, o_win), gates)
        return self.out(heads.transpose(1, 2).reshape(b, t, d))
`;export{e as default};
