const e=`# Gemma 3 270M, the model in the Local–global tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 262144
    d_model: int = 640
    n_layers: int = 18
    n_heads: int = 4
    n_kv_heads: int = 1  # multi-query attention: the four query heads share one key/value head
    head_dim: int = 256
    d_mlp: int = 2048
    sliding_window: int = 512  # a local block's query i sees keys i − 511 … i
    global_every: int = 6  # blocks 6, 12 and 18 are global; the other 15 are local
    rope_local_theta: float = 10_000.0
    rope_global_theta: float = 1_000_000.0
    query_pre_attn_scalar: float = 256.0  # scores are scaled by 1/√256
    eps: float = 1e-6

    def kind(self, layer: int) -> str:
        return "global" if (layer + 1) % self.global_every == 0 else "local"


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    """Gemma's RMSNorm scales by (1 + γ), so a zero weight leaves the normalized vector as it is."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        return (1 + self.weight) * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token lookup times √d_model (√640 ≈ 25.3). Positions enter later, through RoPE."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.scale = c.d_model**0.5

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids) * self.scale  # (batch, seq, d_model)


# %% [rope] RoPE
def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d).
    Local blocks use theta = 10,000; global blocks use 1,000,000, which turns slower over long distances."""
    d = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = positions[:, None].float() * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [sliding-window] Local and global masks
def visible(queries, keys, kind, window):
    """Which keys each query may attend to: every earlier key in a global block, only the last
    \`window\` (itself included) in a local one. queries, keys: positions → (queries, keys) booleans."""
    distance = queries[:, None] - keys[None, :]
    if kind == "global":
        return distance >= 0
    return (distance >= 0) & (distance < window)


# %% [attention] Local / global self-attention
class Attention(nn.Module):
    """Multi-query attention with QK-norm: Q and K pass an RMSNorm of their own before RoPE."""

    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.c, self.layer, self.kind = c, layer, c.kind(layer)
        self.theta = c.rope_global_theta if self.kind == "global" else c.rope_local_theta
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.eps)
        self.k_norm = RMSNorm(c.head_dim, c.eps)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def forward(self, x, start=0, cache=None):  # x: (batch, seq, d_model), starting at position \`start\`
        c, (b, t, _) = self.c, x.shape
        positions = torch.arange(start, start + t, device=x.device)
        q = self.q(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = self.k(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = rope(self.q_norm(q), positions, self.theta), rope(self.k_norm(k), positions, self.theta)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)  # the cached tokens' keys and values, then these
        keys = torch.arange(start + t - k.shape[2], start + t, device=x.device)
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) * c.query_pre_attn_scalar**-0.5
        mask = visible(positions, keys, self.kind, c.sliding_window)
        weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [mlp] MLP
class MLP(nn.Module):
    """GeGLU: down(GELU_tanh(gate(x)) ⊙ up(x)), 640 → 2 × 2,048 → 640."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


# %% [block] Transformer block
class Block(nn.Module):
    """Sandwich norms: each branch normalizes its input and also its output before the residual add."""

    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.pre_attention_norm = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c, layer)
        self.post_attention_norm = RMSNorm(c.d_model, c.eps)
        self.pre_mlp_norm = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)
        self.post_mlp_norm = RMSNorm(c.d_model, c.eps)

    def forward(self, x, start=0, cache=None):
        x = x + self.post_attention_norm(self.attention(self.pre_attention_norm(x), start, cache))
        return x + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))


# %% [kv-cache] KV cache
class KVCache:
    """Keys and values of the tokens seen so far, per block. A global block keeps every token; a
    local block keeps only the last window − 1 (511), because the next query brings its own key."""

    def __init__(self, c: Config):
        self.keep = [None if c.kind(layer) == "global" else c.sliding_window - 1 for layer in range(c.n_layers)]
        self.keys = [None] * c.n_layers  # each (batch, kv_heads, cached tokens, head_dim)
        self.values = [None] * c.n_layers
        self.seen = 0  # tokens processed, so the next one is at position \`seen\`

    def update(self, layer, k, v):
        if self.keys[layer] is not None:
            k = torch.cat((self.keys[layer], k), dim=2)
            v = torch.cat((self.values[layer], v), dim=2)
        keep = self.keep[layer]
        self.keys[layer] = k if keep is None else k[:, :, -keep:]
        self.values[layer] = v if keep is None else v[:, :, -keep:]
        return k, v  # this step attends to all of them, including tokens about to leave the window


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm, then the vocabulary projection, which shares the token embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(x))  # logits: (batch, seq, vocab)


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
class Gemma3(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c, layer) for layer in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, ids, cache=None):  # pass a KVCache to decode token by token
        start = cache.seen if cache is not None else 0
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, start, cache)
        if cache is not None:
            cache.seen += ids.shape[1]
        return self.output(x)


# probabilities = next_token_probabilities(Gemma3()(ids))
`;export{e as default};
