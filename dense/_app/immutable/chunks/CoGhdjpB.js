const n=`# FlashAttention's forward pass (FlashAttention-2, Dao 2023), as the Attention Explainer's Flash tab
# runs it on GPT-2 small's queries, keys and values. Each "# %% [id]" section is one step of the
# diagram. This is the algorithm in plain PyTorch; the real kernel runs the same loop in on-chip SRAM.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12
    block_rows: int = 4  # Br query rows per tile (kernels use 64-128)
    block_cols: int = 4  # Bc keys per tile


# %% [standard] Standard attention: the T × T matrix in memory
def standard_attention(q, k, v):
    """What FlashAttention must reproduce. q, k, v: (batch, heads, seq, head_dim)."""
    t = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (batch, heads, seq, seq)
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
    return weights @ v


# %% [tiles] Score one Br × Bc tile
def score_tile(q_tile, k_tile, rows, cols, scale):
    """Scores of Br queries against Bc keys; inside a diagonal tile, later keys are masked."""
    s = q_tile @ k_tile.transpose(-2, -1) * scale  # (..., Br, Bc)
    return s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))


# %% [online-softmax] Online softmax: fold one tile into the running m, ℓ and output
def online_softmax_step(m, l, acc, s, v_tile):
    """m: running row max, l: running sum of exp(s - m), acc: unnormalized output."""
    m_new = torch.maximum(m, s.amax(dim=-1))
    p = torch.exp(s - m_new[..., None])  # this tile's weights, relative to the new max
    alpha = torch.exp(m - m_new)  # shrinks what earlier tiles added when the max grows
    l = alpha * l + p.sum(dim=-1)
    acc = alpha[..., None] * acc + p @ v_tile
    return m_new, l, acc


# %% [attention] FlashAttention forward: tile by tile, never storing T × T
def flash_attention(q, k, v, c: Config = Config()):
    """Returns the output and the log-sum-exp per row, which the backward pass reuses."""
    *batch, t, d = q.shape
    scale = 1 / math.sqrt(d)
    out = torch.empty_like(q)
    lse = q.new_empty(*batch, t)
    for r0 in range(0, t, c.block_rows):
        rows = torch.arange(r0, min(r0 + c.block_rows, t), device=q.device)
        m = q.new_full((*batch, len(rows)), float("-inf"))
        l = q.new_zeros(*batch, len(rows))
        acc = q.new_zeros(*batch, len(rows), d)
        # Causal: tiles entirely right of the diagonal are skipped, not computed and masked.
        for c0 in range(0, int(rows[-1]) + 1, c.block_cols):
            cols = torch.arange(c0, min(c0 + c.block_cols, t), device=q.device)
            s = score_tile(q[..., rows, :], k[..., cols, :], rows, cols, scale)
            m, l, acc = online_softmax_step(m, l, acc, s, v[..., cols, :])
        out[..., rows, :] = acc / l[..., None]  # normalize once, at the end
        lse[..., rows] = m + torch.log(l)
    return out, lse


# %% [model] GPT-2 attention layer with the FlashAttention forward
class FlashGPT2Attention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.c_attn = nn.Linear(c.d_model, 3 * c.d_model)  # Q, K, V for every head
        self.c_proj = nn.Linear(c.d_model, c.d_model)  # W_O

    def forward(self, x):  # x: (batch, seq, d_model), after ln_1
        b, t, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.c.n_heads, -1).transpose(1, 2) for z in (q, k, v))
        heads, _ = flash_attention(q, k, v, self.c)
        return self.c_proj(heads.transpose(1, 2).reshape(b, t, d))
`;export{n as default};
