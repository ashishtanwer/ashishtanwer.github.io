const e=`# SPLADE (Formal et al., 2021) as opensearch-project/opensearch-neural-sparse-encoding-v2-distill runs it, a 6-layer
# DistilBERT (67 M parameters, Apache-2.0): the Retrieval Explainer's SPLADE tab as plain PyTorch. A text becomes one
# weight per vocabulary term. The masked-language-model head scores all 30,522 WordPiece terms at every token; each
# score is passed through log(1 + ReLU(·)) and the text keeps, per term, its largest value over the tokens. Most terms
# end at exactly 0, so the vector is sparse and can live in an inverted index like BM25's. Query and paragraph go
# through the same model, and their score is the dot product of the two vectors.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 30522
    hidden: int = 768
    layers: int = 6
    heads: int = 12
    intermediate: int = 3072
    max_positions: int = 512
    eps: float = 1e-12
    special_ids: tuple = (0, 100, 101, 102, 103)  # [PAD] [UNK] [CLS] [SEP] [MASK]: the model card zeroes their weights


# %% [embedding] Token and position embeddings
class Embeddings(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.word = nn.Embedding(c.vocab_size, c.hidden)
        self.position = nn.Embedding(c.max_positions, c.hidden)
        self.norm = nn.LayerNorm(c.hidden, eps=c.eps)

    def forward(self, ids):  # ids: (batch, seq); DistilBERT has no segment embeddings
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.norm(self.word(ids) + self.position(positions))


# %% [attention] Self-attention over the unpadded tokens
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
        scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))  # no token reads padding
        weights = scores.softmax(-1)  # (batch, heads, seq, seq)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d)), weights


# %% [mlp] Feed-forward
class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.hidden, c.intermediate)
        self.down = nn.Linear(c.intermediate, c.hidden)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))  # exact (erf) GELU


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


# %% [model] The encoder
class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embeddings = Embeddings(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))

    def forward(self, ids, mask):
        """Last hidden state of every token: (batch, seq, hidden)."""
        x = self.embeddings(ids)
        for block in self.blocks:
            x, _ = block(x, mask)
        return x


# %% [mlm-head] A score for every vocabulary term at every token
class MLMHead(nn.Module):
    """BERT's masked-language-model head, kept from pretraining: dense, GELU, LayerNorm, then a projection onto the
    vocabulary that reuses the token embedding matrix (tied weights) plus a bias per term."""

    def __init__(self, c: Config):
        super().__init__()
        self.dense = nn.Linear(c.hidden, c.hidden)
        self.norm = nn.LayerNorm(c.hidden, eps=c.eps)
        self.bias = nn.Parameter(torch.zeros(c.vocab_size))

    def forward(self, hidden, word):  # hidden: (batch, seq, hidden); word: the (vocab, hidden) embedding matrix
        return self.norm(F.gelu(self.dense(hidden))) @ word.T + self.bias  # (batch, seq, vocab) logits


class Splade(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.encoder = Encoder(c)
        self.head = MLMHead(c)

    def logits(self, ids, mask):
        return self.head(self.encoder(ids, mask), self.encoder.embeddings.word.weight)

    def forward(self, ids, mask):
        """(batch, vocab) sparse vectors: one weight per vocabulary term."""
        return pool(self.logits(ids, mask), mask, self.c)


# %% [pooling] Log saturation and max pooling
def saturate(logits):
    """log(1 + ReLU(logit)): a negative score becomes exactly 0, a large one grows only logarithmically, so no single
    term dominates the dot product."""
    return torch.log1p(torch.relu(logits))


def pool(logits, mask, c: Config):
    """Each term's largest saturated score over the text's tokens (padding excluded), with the special tokens' terms
    set to 0 as the model card does. A term can be non-zero without appearing in the text: an expansion term the
    head predicts from context."""
    weights = saturate(logits * mask[..., None])  # (batch, seq, vocab); padded rows become 0
    vector = weights.max(dim=1).values  # (batch, vocab)
    vector[:, list(c.special_ids)] = 0
    return vector


# %% [score] Dot product of two sparse vectors
def score(queries, docs):
    """(queries, vocab) · (docs, vocab) → (queries, docs). Only terms non-zero in both add anything, so a search
    engine walks the query's terms' posting lists, as for BM25."""
    return queries @ docs.T


def rank(scores, k: int):
    """The k best documents per query, best first: (values, indices)."""
    return scores.topk(k, dim=-1)


# %% [sparsity] How sparse the vectors are
def sparsity(queries, docs):
    """Non-zero terms per query and per paragraph, and SPLADE's FLOPS: the expected number of terms a query and a
    paragraph share, Σ_t P(query has t) · P(paragraph has t), the multiplications one dot product costs in an
    inverted index."""
    q_on, d_on = queries > 0, docs > 0
    flops = (q_on.float().mean(0) * d_on.float().mean(0)).sum()
    return q_on.sum(-1), d_on.sum(-1), flops


# %% [metrics] Recall, MRR and nDCG against the qrels
def metrics(ranked, relevant, ks=(1, 5, 10, 100)):
    """ranked: (queries, depth) document ids, best first; relevant: (queries,) the one relevant id per query."""
    hit = ranked == relevant[:, None]
    found = hit.any(-1)
    rank1 = hit.float().argmax(-1) + 1
    out = {f"recall@{k}": (hit[:, :k].any(-1)).float().mean() for k in ks}
    out["mrr@10"] = torch.where(found & (rank1 <= 10), 1.0 / rank1, 0.0).mean()
    out["ndcg@10"] = torch.where(found & (rank1 <= 10), 1.0 / torch.log2(rank1 + 1.0), 0.0).mean()
    return out
`;export{e as default};
