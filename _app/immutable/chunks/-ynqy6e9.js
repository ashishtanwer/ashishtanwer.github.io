const e=`# BAAI/bge-small-en-v1.5, a 12-layer BERT sentence encoder (33 M parameters): the Retrieval Explainer's Embeddings
# tab as plain PyTorch. A text becomes one unit vector: the [CLS] token's last hidden state, L2-normalized. Search
# ranks every paragraph by the cosine between its vector and the query's; the model was trained with the
# contrastive InfoNCE loss, each query against its own paragraph and every other paragraph in the batch.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 30522
    hidden: int = 384
    layers: int = 12
    heads: int = 12
    intermediate: int = 1536
    max_positions: int = 512
    type_vocab: int = 2
    eps: float = 1e-12
    temperature: float = 0.02  # FlagEmbedding's fine-tuning temperature τ
    query_instruction: str = "Represent this sentence for searching relevant passages: "


# %% [embedding] Token, position and segment embeddings
class Embeddings(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.word = nn.Embedding(c.vocab_size, c.hidden)
        self.position = nn.Embedding(c.max_positions, c.hidden)
        self.segment = nn.Embedding(c.type_vocab, c.hidden)
        self.norm = nn.LayerNorm(c.hidden, eps=c.eps)

    def forward(self, ids):  # ids: (batch, seq)
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.norm(self.word(ids) + self.position(positions) + self.segment(torch.zeros_like(ids)))


# %% [attention] Self-attention over the unpadded tokens
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.q, self.k, self.v = (nn.Linear(c.hidden, c.hidden) for _ in range(3))
        self.out = nn.Linear(c.hidden, c.hidden)

    def forward(self, x, mask):  # x: (batch, seq, hidden); mask: (batch, seq), 1 = token, 0 = padding
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)  # (batch, heads, seq, 32)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))  # no token reads padding
        weights = scores.softmax(-1)  # (batch, heads, seq, seq); every row sums to 1
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d)), weights


# %% [mlp] Feed-forward
class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.hidden, c.intermediate)
        self.down = nn.Linear(c.intermediate, c.hidden)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))  # exact (erf) GELU, as BERT


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
        x = self.norm1(x + a)  # BERT normalizes after each residual sum
        return self.norm2(x + self.mlp(x)), weights


# %% [model] The encoder
class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embeddings = Embeddings(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))

    def forward(self, ids, mask):
        """Last hidden state of every token (batch, seq, hidden) and each block's attention weights."""
        x, weights = self.embeddings(ids), []
        for block in self.blocks:
            x, w = block(x, mask)
            weights.append(w)
        return x, weights


# %% [pooling] One unit vector per text
def embed(encoder: Encoder, ids, mask):
    """[CLS] pooling, then L2 normalization: (batch, hidden) vectors of length 1. The queries are encoded with
    Config.query_instruction in front of the question; paragraphs as they are."""
    hidden, _ = encoder(ids, mask)
    cls = hidden[:, 0]  # the [CLS] token's last hidden state; attention has mixed the whole text into it
    return cls / cls.norm(dim=-1, keepdim=True)


# %% [similarity] Cosine similarity and ranking
def similarity(queries, docs):
    """Unit vectors in, cosines out: (queries, docs). For unit vectors the cosine is the plain dot product."""
    return queries @ docs.T


def rank(scores, k: int):
    """The k best documents per query, best first: (values, indices)."""
    return scores.topk(k, dim=-1)


# %% [metrics] Recall, MRR and nDCG against the qrels
def metrics(ranked, relevant, ks=(1, 5, 10, 100)):
    """ranked: (queries, depth) document ids, best first; relevant: (queries,) the one relevant id per query (SQuAD
    has one). Returns the mean of each metric over the queries."""
    hit = ranked == relevant[:, None]  # (queries, depth), at most one True per row
    found = hit.any(-1)
    rank1 = hit.float().argmax(-1) + 1  # 1-based rank of the relevant document where it was found
    out = {f"recall@{k}": (hit[:, :k].any(-1)).float().mean() for k in ks}
    out["mrr@10"] = torch.where(found & (rank1 <= 10), 1.0 / rank1, 0.0).mean()
    # nDCG@10 with one relevant document: DCG = 1 / log2(rank + 1), and the ideal DCG is 1.
    out["ndcg@10"] = torch.where(found & (rank1 <= 10), 1.0 / torch.log2(rank1 + 1.0), 0.0).mean()
    return out


# %% [infonce] The contrastive InfoNCE loss on an in-batch set
def info_nce(queries, docs, temperature: float):
    """queries[i] belongs with docs[i]; every other doc in the batch is a negative. Row i of the (batch, batch) cosine
    matrix, divided by τ, is a softmax over the batch's documents; the loss is −log of the diagonal's probability."""
    logits = similarity(queries, docs) / temperature
    labels = torch.arange(len(queries), device=queries.device)  # the positives sit on the diagonal
    return F.cross_entropy(logits, labels), logits.softmax(-1)
`;export{e as default};
