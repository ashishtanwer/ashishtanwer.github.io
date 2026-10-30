const e=`# mixedbread-ai/mxbai-embed-large-v1, a 24-layer BERT-large sentence encoder (335 M parameters) trained with
# Matryoshka representation learning: the Retrieval Explainer's Matryoshka tab as plain PyTorch. Its 1024-d vector is
# built so that the first m numbers are a usable vector on their own, so a vector store can keep only 512, 256, 128 or
# 64 of them. Each kept number can also be stored in fewer bits: int8 (one byte, by per-dimension ranges) or one bit
# (its sign), searched by Hamming distance and optionally rescored with the full-precision query.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 30522
    hidden: int = 1024
    layers: int = 24
    heads: int = 16
    intermediate: int = 4096
    max_positions: int = 512
    type_vocab: int = 2
    eps: float = 1e-12
    dims: tuple = (1024, 512, 256, 128, 64)
    query_prompt: str = "Represent this sentence for searching relevant passages: "


# %% [embedding] Token, position and segment embeddings
class Embeddings(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.word = nn.Embedding(c.vocab_size, c.hidden)
        self.position = nn.Embedding(c.max_positions, c.hidden)
        self.segment = nn.Embedding(c.type_vocab, c.hidden)
        self.norm = nn.LayerNorm(c.hidden, eps=c.eps)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.norm(self.word(ids) + self.position(positions) + self.segment(torch.zeros_like(ids)))


# %% [attention] Self-attention
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.q, self.k, self.v = (nn.Linear(c.hidden, c.hidden) for _ in range(3))
        self.out = nn.Linear(c.hidden, c.hidden)

    def forward(self, x, mask):
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)  # (batch, 16, seq, 64)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
        weights = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf")).softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d)), weights


# %% [mlp] Feed-forward
class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.hidden, c.intermediate)
        self.down = nn.Linear(c.intermediate, c.hidden)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


# %% [block] One encoder block (post-norm)
class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attention = Attention(c)
        self.norm1 = nn.LayerNorm(c.hidden, eps=c.eps)
        self.mlp = FeedForward(c)
        self.norm2 = nn.LayerNorm(c.hidden, eps=c.eps)

    def forward(self, x, mask):
        a, _ = self.attention(x, mask)
        x = self.norm1(x + a)
        return self.norm2(x + self.mlp(x))


# %% [model] The encoder and its [CLS] vector
class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embeddings = Embeddings(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))

    def forward(self, ids, mask):
        """The [CLS] token's last hidden state, (batch, 1024): the full Matryoshka vector, not yet normalized."""
        x = self.embeddings(ids)
        for block in self.blocks:
            x = block(x, mask)
        return x[:, 0]


# %% [truncate] Keeping the first m dimensions
def truncate(vectors, m: int):
    """The first m numbers, renormalized to length 1, so the cosine is again a dot product. Matryoshka training adds
    a contrastive loss on prefixes of several lengths, which pushes the most useful information to the front."""
    return F.normalize(vectors[:, :m], dim=-1)


# %% [int8] One byte per dimension
def int8_ranges(corpus):
    """Per-dimension (start, step) from the corpus' own minimum and maximum: 256 buckets between them."""
    start, stop = corpus.min(0).values, corpus.max(0).values
    step = (stop - start) / 255
    return start, torch.where(step == 0, torch.ones_like(step), step)


def int8_codes(vectors, start, step):
    """Each number becomes its bucket, −128 … 127 (values outside the corpus range land in the end buckets)."""
    return (torch.floor((vectors - start) / step).clamp(0, 255) - 128).to(torch.int8)


def int8_values(codes, start, step):
    """Back to numbers: the middle of each bucket. Search compares the full-precision query with these."""
    return (codes.float() + 128 + 0.5) * step + start


# %% [binary] One bit per dimension and Hamming distance
def binary_codes(vectors):
    """The sign of each number, packed 8 to a byte (most significant bit first): m / 8 bytes per vector."""
    bits = (vectors > 0).to(torch.uint8).view(len(vectors), -1, 8)
    return (bits << torch.arange(7, -1, -1, dtype=torch.uint8)).sum(-1, dtype=torch.uint8)


def hamming(a, b):
    """Differing bits between every packed vector of a and of b, (len(a), len(b)): fewer = more similar."""
    x = a[:, None, :] ^ b[None, :, :]
    return ((x[..., None] >> torch.arange(8, dtype=torch.uint8)) & 1).sum((-1, -2))


# %% [rescore] Rescoring the binary search's candidates
def rescore(query, candidates, bits):
    """Binary search, then a second look: the full-precision query dotted with the candidates' bits (0 or 1)."""
    unpacked = ((bits[candidates][..., None] >> torch.arange(7, -1, -1, dtype=torch.uint8)) & 1).flatten(-2)
    return torch.einsum("qd,qkd->qk", query, unpacked.float())
`;export{e as default};
