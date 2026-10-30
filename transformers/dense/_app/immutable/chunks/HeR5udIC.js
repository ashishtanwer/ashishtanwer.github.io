const e=`# DeepSeek-V2-Lite, the model in the MLA tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 102400
    d_model: int = 2048
    n_layers: int = 27
    n_heads: int = 16
    d_content: int = 128  # per-head query and key dimensions that carry no position
    d_rotary: int = 64  # per-head query dimensions RoPE rotates; the 64-D rotary key is shared by all heads
    d_value: int = 128
    d_latent: int = 512  # the compressed KV; with the rotary key, 576 numbers per token in the cache
    d_mlp: int = 10944  # the dense MLP of block 1
    n_dense_layers: int = 1
    n_experts: int = 64
    n_shared_experts: int = 2
    top_k: int = 6
    d_expert: int = 1408
    rope_theta: float = 10000.0
    # YaRN stretches RoPE from the 4,096 positions it was trained on to 163,840.
    rope_factor: float = 40.0
    rope_original_positions: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 0.707
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
    """Token lookup. Positions enter later, through RoPE on the rotary dimensions."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids)  # (batch, seq, d_model)


# %% [rope] RoPE
def yarn_frequencies(c: Config):
    """One frequency per pair of the 64 rotary dimensions, stretched by YaRN.

    Pairs that turn more than beta_fast times over the original 4,096 positions keep their
    frequency; pairs that turn less than beta_slow times are slowed by rope_factor; the pairs in
    between blend the two."""
    d = c.d_rotary
    frequencies = c.rope_theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)  # (d/2,)

    def pair(turns):  # the pair whose wavelength fits \`turns\` times into the original context
        return d * math.log(c.rope_original_positions / (turns * 2 * math.pi)) / (2 * math.log(c.rope_theta))

    low, high = max(math.floor(pair(c.rope_beta_fast)), 0), min(math.ceil(pair(c.rope_beta_slow)), d - 1)
    slowed = ((torch.arange(d // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
    return frequencies / c.rope_factor * slowed + frequencies * (1 - slowed)


def rope(x, frequencies):
    """Rotate each pair (2i, 2i + 1) of the last dimension by position × frequency i.

    DeepSeek's own code first reorders the 64 dimensions to (evens, odds) and rotates halves;
    queries and keys are reordered alike, so the attention scores are the same."""
    angles = torch.arange(x.shape[-2], dtype=torch.float32)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


# %% [query] Query
class Query(nn.Module):
    """One projection gives every head a 128-D content query and a 64-D query that RoPE rotates."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.proj = nn.Linear(c.d_model, c.n_heads * (c.d_content + c.d_rotary), bias=False)  # Wq: 2048 → 3072

    def forward(self, x, frequencies):  # x: (batch, seq, d_model)
        b, t, _ = x.shape
        q = self.proj(x).view(b, t, self.c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, 192)
        content, rotary = q.split([self.c.d_content, self.c.d_rotary], dim=-1)
        return content, rope(rotary, frequencies)


# %% [latent-kv] Latent KV · cache
class LatentKV(nn.Module):
    """W↓KV compresses each token to a 512-D latent and a 64-D key that RoPE rotates. Those 576
    numbers per token are all the cache keeps; W↑K and W↑V expand the normalized latent into every
    head's own content key and value."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.down = nn.Linear(c.d_model, c.d_latent + c.d_rotary, bias=False)  # W↓KV: 2048 → 576
        self.norm = RMSNorm(c.d_latent, c.eps)
        # W↑K and W↑V in one matrix, 512 → 16 heads × (128 key + 128 value)
        self.up = nn.Linear(c.d_latent, c.n_heads * (c.d_content + c.d_value), bias=False)

    def compress(self, x, frequencies):
        """The cache entry of each token: (batch, seq, 512) latent and (batch, 1, seq, 64) rotated key."""
        latent, rotary_key = self.down(x).split([self.c.d_latent, self.c.d_rotary], dim=-1)
        return self.norm(latent), rope(rotary_key[:, None], frequencies)

    def expand(self, latent):
        """Every head's content keys and values, (batch, heads, seq, 128) each, from the cached latent."""
        b, t, _ = latent.shape
        kv = self.up(latent).view(b, t, self.c.n_heads, -1).transpose(1, 2)
        return kv.split([self.c.d_content, self.c.d_value], dim=-1)


# %% [attention] Multi-head latent attention
def softmax_scale(c: Config):
    """1/√192 times YaRN's mscale², (0.1 × 0.707 × ln 40 + 1)²: 0.1147 for DeepSeek-V2-Lite."""
    mscale = 0.1 * c.rope_mscale * math.log(c.rope_factor) + 1
    return (c.d_content + c.d_rotary) ** -0.5 * mscale**2


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.query = Query(c)
        self.kv = LatentKV(c)
        self.out = nn.Linear(c.n_heads * c.d_value, c.d_model, bias=False)
        self.scale = softmax_scale(c)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, _ = x.shape
        frequencies = yarn_frequencies(self.c)
        q_content, q_rotary = self.query(x, frequencies)
        latent, k_rotary = self.kv.compress(x, frequencies)  # what the cache would store
        k_content, v = self.kv.expand(latent)
        # Content scores use each head's own key; rotary scores use the one key all heads share.
        scores = q_content @ k_content.transpose(-2, -1) + q_rotary @ k_rotary.transpose(-2, -1)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = (scores * self.scale).masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [mlp] Mixture of experts
class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)). Block 1 has one 10,944-wide MLP; each expert is 1,408 wide."""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """A softmax router picks 6 of 64 experts per token and weights each by its router probability,
    without renormalizing. Two shared experts, run as one MLP of twice the width, see every token."""

    def __init__(self, c: Config):
        super().__init__()
        self.top_k = c.top_k
        self.router = nn.Linear(c.d_model, c.n_experts, bias=False)
        self.experts = nn.ModuleList(MLP(c.d_model, c.d_expert) for _ in range(c.n_experts))
        self.shared = MLP(c.d_model, c.n_shared_experts * c.d_expert)

    def forward(self, x):  # x: (batch, seq, d_model)
        tokens = x.reshape(-1, x.shape[-1])
        probabilities = self.router(tokens).softmax(dim=-1)  # over all 64 experts
        weights, experts = probabilities.topk(self.top_k, dim=-1)  # kept as they are: they sum to less than 1
        routed = torch.zeros_like(tokens)
        for n, token in enumerate(tokens):
            for weight, e in zip(weights[n], experts[n]):
                routed[n] += weight * self.experts[e](token)
        return (routed + self.shared(tokens)).view(x.shape)


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(RMSNorm(x)), then x + MLP(RMSNorm(x))."""

    def __init__(self, c: Config, index: int):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c.d_model, c.d_mlp) if index < c.n_dense_layers else MoE(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm, then a projection to one logit per vocabulary token (not tied to the embedding)."""

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
class DeepSeekV2(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c, i) for i in range(c.n_layers))
        self.output = Output(c)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# probabilities = next_token_probabilities(DeepSeekV2()(ids), temperature=0.8, top_k=5)
`;export{e as default};
