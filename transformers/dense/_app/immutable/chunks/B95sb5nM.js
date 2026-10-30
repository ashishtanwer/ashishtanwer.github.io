const e=`# Granite 3.1 1B-A400M, the model in the MoE tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 49152
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16  # 64 dimensions per head
    n_kv_heads: int = 8  # grouped-query attention: two query heads share each key/value head
    n_experts: int = 32
    top_k: int = 8  # experts each token is routed to
    d_expert: int = 512
    rope_theta: float = 1_500_000.0
    embedding_multiplier: float = 12.0
    attention_multiplier: float = 0.015625  # used instead of 1 / sqrt(head_dim)
    residual_multiplier: float = 0.22
    logits_scaling: float = 6.0
    eps: float = 1e-6


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token lookup times a constant. Positions enter later, through RoPE inside attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.multiplier = c.embedding_multiplier

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids) * self.multiplier


# %% [rope] RoPE
def rope(x, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    seq, d = x.shape[-2], x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(seq, device=x.device)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Grouped-query self-attention
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * head_dim, c.d_model, bias=False)

    def forward(self, x):  # x: (batch, seq, d_model)
        c, (b, t, _) = self.c, x.shape
        q = self.q(x).view(b, t, c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = self.k(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) * c.attention_multiplier
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [mlp] Mixture of experts
class MoE(nn.Module):
    """A router sends each token to its top-k experts; each expert is a small SwiGLU MLP."""

    def __init__(self, c: Config):
        super().__init__()
        self.top_k = c.top_k
        self.router = nn.Linear(c.d_model, c.n_experts, bias=False)
        # Every expert's gate and up projections, stacked: (experts, 2 × d_expert, d_model)
        self.up = nn.Parameter(torch.empty(c.n_experts, 2 * c.d_expert, c.d_model))
        self.down = nn.Parameter(torch.empty(c.n_experts, c.d_model, c.d_expert))

    def forward(self, x):  # x: (batch, seq, d_model)
        tokens = x.reshape(-1, x.shape[-1])
        top_logits, experts = self.router(tokens).topk(self.top_k, dim=-1)
        gates = top_logits.softmax(dim=-1)  # weights over the chosen experts only
        out = torch.zeros_like(tokens)
        for n, token in enumerate(tokens):
            for gate, e in zip(gates[n], experts[n]):
                g, u = (self.up[e] @ token).chunk(2)
                out[n] += gate * (self.down[e] @ (F.silu(g) * u))
        return out.view(x.shape)


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm block whose residual branches are scaled by residual_multiplier."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.moe = MoE(c)
        self.residual_multiplier = c.residual_multiplier

    def forward(self, x):
        x = x + self.residual_multiplier * self.attention(self.norm1(x))
        return x + self.residual_multiplier * self.moe(self.norm2(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm, the vocabulary projection, then division by logits_scaling."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding
        self.logits_scaling = c.logits_scaling

    def forward(self, x):
        return self.lm_head(self.norm(x)) / self.logits_scaling  # logits: (batch, seq, vocab)


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
class GraniteMoE(nn.Module):
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


# probabilities = next_token_probabilities(GraniteMoE()(ids), temperature=0.8, top_k=5)
`;export{e as default};
