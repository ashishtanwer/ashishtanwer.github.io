const e=`# BLT (facebook/blt-1b, Meta FAIR): no vocabulary of subwords. A small byte model predicts each next byte; where its
# prediction is uncertain (high entropy) a new patch starts. The local encoder turns each patch's bytes into one
# vector for the big global transformer. Weights: FAIR Noncommercial Research License.
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Config:
    bos_id: int = 1  # <s>; 0 is <boe>, 2 is </s>, 3 is <pad>
    offset: int = 4  # byte b has id b + 4
    vocab_size: int = 260
    # the entropy model (facebook/blt-entropy, 100 M parameters)
    entropy_dim: int = 768
    entropy_layers: int = 14
    entropy_heads: int = 12
    entropy_ffn: int = 2048
    entropy_rope_theta: float = 10000.0
    threshold: float = 1.335442066192627  # nats; global rule, no monotonicity (blt-1b's config)
    # the local encoder of the 1B model
    dim: int = 1024
    heads: int = 16
    ffn: int = 2816
    encoder_layers: int = 1
    rope_theta: float = 500000.0
    cross_attn_k: int = 2  # each patch becomes k vectors of dim, i.e. one 2048-wide vector for the global model
    hash_sizes: tuple = (3, 4, 5, 6, 7, 8)  # byte n-grams hashed into their own embedding tables
    hash_vocab: int = 500002
    hash_prime: int = 1000000007
    eps: float = 1e-5


# %% [ids] Byte ids
def encode(text: str, config: Config = Config()) -> list[int]:
    """<s>, then one id per UTF-8 byte: byte + 4."""
    return [config.bos_id] + [b + config.offset for b in text.encode('utf-8')]


# %% [entropy] Entropy model
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(dim)), eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rope(x, theta):
    """Rotary positions on (batch, heads, length, head_dim), rotating neighbouring pairs (0, 1), (2, 3), …"""
    length, half = x.shape[-2], x.shape[-1] // 2
    freqs = torch.arange(length)[:, None].float() / theta ** (torch.arange(half).float() / half)
    cos, sin = freqs.cos().repeat_interleave(2, -1), freqs.sin().repeat_interleave(2, -1)
    pairs = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
    return x * cos + pairs * sin


class Layer(nn.Module):
    """Pre-norm block: causal self-attention with RoPE, then a SwiGLU MLP."""

    def __init__(self, dim, heads, ffn, theta, eps):
        super().__init__()
        self.heads, self.theta = heads, theta
        self.attention_norm, self.ffn_norm = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim, bias=False) for _ in range(4))
        self.w1, self.w3 = nn.Linear(dim, ffn, bias=False), nn.Linear(dim, ffn, bias=False)
        self.w2 = nn.Linear(ffn, dim, bias=False)

    def forward(self, x):  # (batch, length, dim)
        b, n, d = x.shape
        h = self.attention_norm(x)
        q, k, v = (w(h).view(b, n, self.heads, -1).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = rope(q, self.theta), rope(k, self.theta)
        scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float('-inf'))
        x = x + self.wo((scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d))
        h = self.ffn_norm(x)
        return x + self.w2(nn.functional.silu(self.w1(h)) * self.w3(h))


class EntropyModel(nn.Module):
    """A small byte-level language model; only its next-byte distributions are used."""

    def __init__(self, config: Config = Config()):
        super().__init__()
        c = config
        self.tok_embeddings = nn.Embedding(c.vocab_size, c.entropy_dim)
        self.layers = nn.ModuleList(
            Layer(c.entropy_dim, c.entropy_heads, c.entropy_ffn, c.entropy_rope_theta, c.eps) for _ in range(c.entropy_layers)
        )
        self.norm = RMSNorm(c.entropy_dim, c.eps)
        self.output = nn.Linear(c.entropy_dim, c.vocab_size, bias=False)

    def forward(self, ids):  # (batch, length) → next-byte logits (batch, length, 260)
        x = self.tok_embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))


def next_byte_entropy(logits):
    """H = −Σ p log p over the 260 ids, in nats: how unsure the model is about the byte after each position."""
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


# %% [patching] Patch boundaries
def patch_lengths(entropy: torch.Tensor, config: Config = Config()) -> list[int]:
    """Positions 0 (<s>) and 1 always start a patch; after that, position p + 1 starts one when the entropy at p
    (the uncertainty about the byte at p + 1) is above the threshold. If the last position's entropy is above it,
    the next, unseen byte would start a patch: an empty patch closes the list."""
    n = entropy.shape[-1]
    starts = [0, 1] + [p + 1 for p in range(1, n) if entropy[p] > config.threshold]
    ends = starts[1:] + [n]
    return [end - start for start, end in zip(starts, ends)]


# %% [hash] Hash n-gram embeddings
def ngram_hashes(ids: torch.Tensor, n: int, config: Config = Config()) -> torch.Tensor:
    """For each position, the n ids ending there (zeros before the start) as one number: a polynomial in the prime,
    computed in int64 (it wraps around, as in BLT's code), then taken modulo the table size."""
    padded = torch.cat([torch.zeros(ids.shape[0], n - 1, dtype=torch.long), ids], dim=1)
    windows = padded.unfold(1, n, 1)  # (batch, length, n)
    powers = torch.tensor(config.hash_prime, dtype=torch.long) ** torch.arange(n)
    return (windows * powers).sum(-1) % config.hash_vocab


class HashEmbedding(nn.Module):
    """A byte's vector: its own embedding plus one row per n-gram size, so the encoder sees the bytes before it."""

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tables = nn.ModuleList(nn.Embedding(config.hash_vocab, config.dim) for _ in config.hash_sizes)

    def forward(self, ids):  # (batch, length) → (batch, length, dim)
        x = self.tok_embeddings(ids)
        for n, table in zip(self.config.hash_sizes, self.tables):
            x = x + table(ngram_hashes(ids, n, self.config))
        return x


# %% [encoder] Local encoder
class CrossAttention(nn.Module):
    """Each patch's k query vectors attend only to the bytes of that patch."""

    def __init__(self, dim, heads, eps):
        super().__init__()
        self.heads = heads
        self.cross_attn_norm_q, self.cross_attn_norm_kv = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim, bias=False) for _ in range(4))

    def forward(self, queries, keys, allowed):  # (b, P·k, dim), (b, length, dim), (P·k, length) bool
        b, m, d = queries.shape
        q = self.wq(self.cross_attn_norm_q(queries)).view(b, m, self.heads, -1).transpose(1, 2)
        kv = self.cross_attn_norm_kv(keys)
        k, v = (w(kv).view(b, kv.shape[1], self.heads, -1).transpose(1, 2) for w in (self.wk, self.wv))
        scores = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).masked_fill(~allowed, torch.finfo(q.dtype).min)
        weights = scores.softmax(-1)  # (b, heads, P·k, length)
        return queries + self.wo((weights @ v).transpose(1, 2).reshape(b, m, d)), weights


class LocalEncoder(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        c = self.config = config
        self.layers = nn.ModuleList(Layer(c.dim, c.heads, c.ffn, c.rope_theta, c.eps) for _ in range(c.encoder_layers))
        self.patch_embedding_projection = nn.Linear(c.dim, c.dim * c.cross_attn_k, bias=False)
        self.cross_attn = CrossAttention(c.dim, c.heads, c.eps)

    def forward(self, x, lengths: list[int]):  # x: (1, length, dim) from HashEmbedding
        for layer in self.layers:
            x = layer(x)
        patch_of = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))  # (length,)
        # each patch starts as the max over its bytes (an empty patch stays 0), projected to k query vectors
        pooled = torch.stack(
            [x[0, patch_of == p].amax(0) if size else x.new_zeros(x.shape[-1]) for p, size in enumerate(lengths)]
        )
        k = self.config.cross_attn_k
        queries = self.patch_embedding_projection(pooled).view(1, len(lengths) * k, -1)
        allowed = torch.arange(len(lengths)).repeat_interleave(k)[:, None] == patch_of[None, :]
        attended, weights = self.cross_attn(queries, x, allowed)
        # BLT adds the queries twice: inside the cross-attention (queries + output) and again here
        patches = queries + attended
        return patches.view(1, len(lengths), -1), x, weights  # patch vectors (1, P, k·dim) for the global model


# %% [model] Text to patch vectors
def blt_patches(text, entropy_model, embed, encoder, config: Config = Config()):
    ids = torch.tensor([encode(text, config)])
    entropy = next_byte_entropy(entropy_model(ids))[0]
    lengths = patch_lengths(entropy, config)
    patches, _, _ = encoder(embed(ids), lengths)
    return entropy, lengths, patches
`;export{e as default};
