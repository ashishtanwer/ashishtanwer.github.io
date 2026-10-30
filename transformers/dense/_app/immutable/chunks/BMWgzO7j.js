const e=`# BAAI/bge-reranker-base, a cross-encoder (XLM-RoBERTa base, 278 M parameters): the Retrieval Explainer's Reranker tab as
# plain PyTorch. Unlike the embedding model, it never makes a vector per text: the query and one paragraph are read
# together as one sequence, every query token attends to every paragraph token, and a small head turns the first
# token's last hidden state into one relevance score. That takes one forward pass per (query, paragraph) pair, so it
# only reorders the few paragraphs a fast embedding search returned.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 250002
    hidden: int = 768
    layers: int = 12
    heads: int = 12
    intermediate: int = 3072
    max_positions: int = 514
    pad_id: int = 1
    bos_id: int = 0  # <s>
    eos_id: int = 2  # </s>
    eps: float = 1e-5
    max_length: int = 512


# %% [pair] Query and paragraph as one input
def pair_ids(query_ids: list[int], doc_ids: list[int], c: Config) -> list[int]:
    """<s> query </s></s> paragraph </s>: XLM-RoBERTa's sentence-pair layout (no segment ids, only the separators).
    Longer pairs lose tokens from the end of the longer text (the tokenizer's longest_first truncation)."""
    q, d = list(query_ids), list(doc_ids)
    while len(q) + len(d) + 4 > c.max_length:
        (d if len(d) >= len(q) else q).pop()
    return [c.bos_id, *q, c.eos_id, c.eos_id, *d, c.eos_id]


# %% [embedding] Token and position embeddings
class Embeddings(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.pad_id = c.pad_id
        self.word = nn.Embedding(c.vocab_size, c.hidden)
        self.position = nn.Embedding(c.max_positions, c.hidden)
        self.segment = nn.Embedding(1, c.hidden)  # one segment: the pair is told apart by </s></s> alone
        self.norm = nn.LayerNorm(c.hidden, eps=c.eps)

    def forward(self, ids):  # ids: (batch, seq)
        real = (ids != self.pad_id).long()
        positions = real.cumsum(1) * real + self.pad_id  # RoBERTa counts positions from pad_id + 1 = 2
        return self.norm(self.word(ids) + self.position(positions) + self.segment(torch.zeros_like(ids)))


# %% [attention] Self-attention over query and paragraph together
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.q, self.k, self.v = (nn.Linear(c.hidden, c.hidden) for _ in range(3))
        self.out = nn.Linear(c.hidden, c.hidden)

    def forward(self, x, mask):  # x: (batch, seq, hidden); mask: (batch, seq), 1 = token, 0 = padding
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)  # (batch, heads, seq, 64)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        weights = scores.softmax(-1)  # query tokens read paragraph tokens and the other way round
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
        a, weights = self.attention(x, mask)
        x = self.norm1(x + a)
        return self.norm2(x + self.mlp(x)), weights


# %% [head] The relevance score
class Head(nn.Module):
    """RoBERTa's classification head on the first token <s>: dense, tanh, then one output = the score (a logit)."""

    def __init__(self, c: Config):
        super().__init__()
        self.dense = nn.Linear(c.hidden, c.hidden)
        self.score = nn.Linear(c.hidden, 1)

    def forward(self, hidden):  # (batch, seq, hidden) → (batch,)
        return self.score(torch.tanh(self.dense(hidden[:, 0]))).squeeze(-1)


# %% [model] The cross-encoder
class CrossEncoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embeddings = Embeddings(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.head = Head(c)

    def forward(self, ids, mask):
        """One relevance logit per (query, paragraph) row of ids, and every block's attention weights."""
        x, weights = self.embeddings(ids), []
        for block in self.blocks:
            x, w = block(x, mask)
            weights.append(w)
        return self.head(x), weights


# %% [rerank] Reordering the first stage's top k
def rerank(first_stage, scores, k: int):
    """first_stage: (queries, depth) paragraph ids from the embedding search, best first; scores: (queries, ≥ k) the
    cross-encoder's logits for the first k of them. The first k are sorted by logit; the rest keep their order."""
    order = scores[:, :k].argsort(dim=-1, descending=True, stable=True)
    return torch.cat([first_stage[:, :k].gather(1, order), first_stage[:, k:]], dim=1)


def probability(logit):
    """bge-reranker's normalized score: sigmoid(logit), a number in (0, 1) for display; the ranking is the same."""
    return torch.sigmoid(logit)
`;export{e as default};
