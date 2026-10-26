const n=`# GPT-2 small, the model in the Dense tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    max_positions: int = 1024
    d_model: int = 768
    n_heads: int = 12  # 768 / 12 = 64 dimensions per head
    n_layers: int = 12
    d_mlp: int = 3072  # 4 × d_model
    eps: float = 1e-5


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token embedding plus a learned position embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.position = nn.Embedding(c.max_positions, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.token(ids) + self.position(positions)  # (batch, seq, d_model)


# %% [attention] Multi-head self-attention
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)  # Q, K and V for every head at once
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # Split into heads: (batch, heads, seq, head_dim)
        q, k, v = (z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        heads = weights @ v  # (batch, heads, seq, head_dim)
        return self.out(heads.transpose(1, 2).reshape(b, t, d))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c)
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final LayerNorm, then a projection back to one logit per vocabulary token."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_f = nn.LayerNorm(c.d_model, eps=c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding

    def forward(self, x):
        return self.lm_head(self.ln_f(x))  # logits: (batch, seq, vocab)


def next_token_probabilities(logits, temperature=1.0, top_k=None, top_p=None):
    """The last position's logits, scaled by temperature and cut to top-k or top-p, as probabilities."""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        p = sorted_logits.softmax(dim=-1)
        drop = p.cumsum(dim=-1) - p >= top_p  # keep the smallest set whose mass reaches top_p
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits.softmax(dim=-1)


# %% [model] Full model
class GPT2(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# probabilities = next_token_probabilities(GPT2()(ids), temperature=0.8, top_k=5)
`;export{n as default};
