const e=`# ModernBERT-base, the model in the Encoder-only tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50368
    d_model: int = 768
    n_layers: int = 22
    n_heads: int = 12  # 64 dimensions per head
    d_mlp: int = 1152  # GeGLU: Wi makes 2 × 1152, Wo reads 1152
    global_every: int = 3  # blocks 1, 4, 7, ... attend globally, the others in a sliding window
    window: int = 128  # a local block's token sees 64 positions on each side
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    mask_id: int = 50284  # [MASK]
    eps: float = 1e-5


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token lookup, then LayerNorm. No position vector is added: RoPE rotates Q and K instead."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps, bias=False)

    def forward(self, ids):  # ids: (batch, seq)
        return self.norm(self.token(ids))  # (batch, seq, d_model)


# %% [rope] RoPE
def rope(x, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    seq, d = x.shape[-2], x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(seq, device=x.device)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Bidirectional self-attention
class Attention(nn.Module):
    """No causal mask. Global blocks see every token; the others see a sliding window and use a
    smaller RoPE base."""

    def __init__(self, c: Config, layer: int):
        super().__init__()
        is_global = layer % c.global_every == 0
        self.n_heads = c.n_heads
        self.window = None if is_global else c.window
        self.theta = c.global_rope_theta if is_global else c.local_rope_theta
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)  # Wqkv: Q, K and V at once
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)  # Wo

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, -1).unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # (batch, heads, seq, head_dim)
        q, k = rope(q, self.theta), rope(k, self.theta)  # values are not rotated
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if self.window is not None:
            position = torch.arange(t, device=x.device)
            far = (position[:, None] - position[None, :]).abs() > self.window // 2
            scores = scores.masked_fill(far, float("-inf"))
        weights = scores.softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, d))


# %% [mlp] MLP
class MLP(nn.Module):
    """GeGLU: Wi makes two halves u and g, and Wo reads GELU(u) ⊙ g."""

    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, 2 * c.d_mlp, bias=False)  # Wi
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)  # Wo

    def forward(self, x):
        u, g = self.up(x).chunk(2, dim=-1)  # (batch, seq, d_mlp) each
        return self.down(F.gelu(u) * g)  # exact (erf) GELU


# %% [block] Encoder block
class Block(nn.Module):
    """Pre-norm residual block. Block 1 has no attention LayerNorm: the embedding ends in one."""

    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.norm1 = nn.Identity() if layer == 0 else nn.LayerNorm(c.d_model, eps=c.eps, bias=False)
        self.attention = Attention(c, layer)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps, bias=False)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [output] Masked-token prediction
class Output(nn.Module):
    """Final LayerNorm, the masked-LM head (Dense, GELU, LayerNorm), then the decoder, which shares the
    token embedding and has its own bias."""

    def __init__(self, c: Config):
        super().__init__()
        self.final_norm = nn.LayerNorm(c.d_model, eps=c.eps, bias=False)
        self.dense = nn.Linear(c.d_model, c.d_model, bias=False)
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps, bias=False)
        self.decoder = nn.Linear(c.d_model, c.vocab_size)

    def forward(self, x):
        h = self.norm(F.gelu(self.dense(self.final_norm(x))))
        return self.decoder(h)  # logits: (batch, seq, vocab)


def mask_probabilities(logits, ids, mask_id, temperature=1.0, top_k=None, top_p=None):
    """The logits at each [MASK], scaled by temperature and cut to top-k or top-p, as probabilities."""
    logits = logits[ids == mask_id] / temperature  # (masks, vocab)
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
class ModernBERT(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c, layer) for layer in range(c.n_layers))
        self.output = Output(c)
        self.output.decoder.weight = self.embedding.token.weight  # weight tying

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# ids = tokenizer("The capital of France is [MASK].")
# probabilities = mask_probabilities(ModernBERT()(ids), ids, Config.mask_id, top_k=5)
`;export{e as default};
