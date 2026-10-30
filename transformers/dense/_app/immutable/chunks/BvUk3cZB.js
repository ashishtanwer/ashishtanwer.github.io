const e=`# Differential attention ("Differential Transformer", Ye et al., ICLR 2025), shaped like GPT-2 small,
# whose recorded Q, K and V the Attention Explainer's Diff tab feeds it: 12 heads, each with two
# 32-D query and key halves and a 64-D value. Follows Microsoft's multihead_diffattn.py, without the
# rotary embedding (GPT-2's Q and K already carry learned positions). Each "# %% [id]" section is
# one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12  # differential heads; half the head count of a standard layer of this width
    depth: int = 5  # 0-based layer index, which sets λ_init


# %% [split] Split each head's query and key into two halves
def split_halves(q, k):
    """q, k: (batch, heads, seq, 2d). Half 1 and half 2 feed two separate softmax maps."""
    d = q.shape[-1] // 2
    return q[..., :d], q[..., d:], k[..., :d], k[..., d:]


# %% [lambda] λ, re-parameterized from four learned vectors
def lambda_init(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class Lambda(nn.Module):
    """λ = exp(λ_q1 · λ_k1) − exp(λ_q2 · λ_k2) + λ_init. The tab sets the four vectors to zero
    (λ = λ_init); a trained model learns them."""

    def __init__(self, head_dim, depth):
        super().__init__()
        self.init = lambda_init(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim))

    def forward(self):
        first = torch.exp((self.lambda_q1 * self.lambda_k1).sum())
        second = torch.exp((self.lambda_q2 * self.lambda_k2).sum())
        return first - second + self.init


# %% [attention] Differential attention: one softmax minus λ times another
def differential_attention(q, k, v, lam):
    """q, k: (batch, heads, seq, 2d); v: (batch, heads, seq, 2d). Returns the mixed values and the
    signed matrix the head card draws; its rows sum to 1 − λ."""
    t = q.shape[-2]
    q1, q2, k1, k2 = split_halves(q, k)
    scale = 1 / math.sqrt(q1.shape[-1])
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    first = (q1 @ k1.transpose(-2, -1) * scale).masked_fill(~causal, float("-inf")).softmax(dim=-1)
    second = (q2 @ k2.transpose(-2, -1) * scale).masked_fill(~causal, float("-inf")).softmax(dim=-1)
    weights = first - lam * second  # noise both maps share cancels
    return weights @ v, weights


# %% [norm] Per-head RMSNorm, scaled by 1 − λ_init
class HeadNorm(nn.Module):
    def __init__(self, dim, depth, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = 1 - lambda_init(depth)
        self.weight = nn.Parameter(torch.ones(dim))  # the tab keeps the initial gain of 1

    def forward(self, x):  # x: (batch, heads, seq, 2d), each head normalized on its own
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight * self.scale


# %% [model] Differential attention layer
class DifferentialAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.n_heads = c.n_heads
        head_dim = c.d_model // c.n_heads // 2  # d: each half of a head's query and key
        self.q_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out_proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.lam = Lambda(head_dim, c.depth)
        self.subln = HeadNorm(2 * head_dim, c.depth)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        q, k, v = (proj(x).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q_proj, self.k_proj, self.v_proj))
        mixed, _ = differential_attention(q, k, v, self.lam())
        heads = self.subln(mixed)
        return self.out_proj(heads.transpose(1, 2).reshape(b, t, d))
`;export{e as default};
