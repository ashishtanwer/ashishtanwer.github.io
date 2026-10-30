const n=`# colbert-ir/colbertv2.0, late interaction over token vectors (BERT-base, 110 M parameters): the Retrieval Explainer's
# ColBERT tab as plain PyTorch. Query and paragraph are encoded separately, as with a sentence embedding, but every
# token keeps its own 128-d unit vector. The score is MaxSim: each query token finds its most similar paragraph token,
# and those best cosines are summed. Paragraph vectors can be computed ahead of time; only the query runs at search
# time, and the interaction is a matrix product.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 30522
    hidden: int = 768
    layers: int = 12
    heads: int = 12
    intermediate: int = 3072
    max_positions: int = 512
    type_vocab: int = 2
    eps: float = 1e-12
    dim: int = 128  # the token vectors' size
    query_maxlen: int = 32
    doc_maxlen: int = 180
    cls_id: int = 101
    sep_id: int = 102
    mask_id: int = 103
    pad_id: int = 0
    query_marker: int = 1  # [unused0], shown as [Q]
    doc_marker: int = 2  # [unused1], shown as [D]


# %% [query-input] The query, padded with [MASK]
def query_input(token_ids: list[int], c: Config):
    """[CLS] [Q] question [SEP], then [MASK] up to 32 tokens. The [MASK]s are not attended to, but each still gets an
    output vector: query augmentation, extra soft query terms the model learned to fill in."""
    body = token_ids[: c.query_maxlen - 3]
    ids = [c.cls_id, c.query_marker, *body, c.sep_id]
    mask = [1] * len(ids) + [0] * (c.query_maxlen - len(ids))
    ids += [c.mask_id] * (c.query_maxlen - len(ids))
    return torch.tensor(ids), torch.tensor(mask)


# %% [doc-input] The paragraph, punctuation left out
def doc_input(token_ids: list[int], skip_ids: set, c: Config):
    """[CLS] [D] paragraph [SEP], at most 180 tokens. \`keep\` drops the punctuation tokens' vectors (they would match
    any query's punctuation) after encoding; they still take part in attention."""
    ids = torch.tensor([c.cls_id, c.doc_marker, *token_ids[: c.doc_maxlen - 3], c.sep_id])
    keep = torch.tensor([i not in skip_ids for i in ids.tolist()])
    return ids, keep


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

    def forward(self, x, mask):  # mask: (batch, seq), 1 = attended, 0 = not (padding, a query's [MASK]s)
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)
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
        a, weights = self.attention(x, mask)
        x = self.norm1(x + a)
        return self.norm2(x + self.mlp(x)), weights


# %% [projection] One 128-d unit vector per token
class ColBERT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embeddings = Embeddings(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.linear = nn.Linear(c.hidden, c.dim, bias=False)  # 768 → 128, ColBERT's only new weights

    def forward(self, ids, mask, keep=None):
        """(batch, seq, 128) unit vectors; rows with keep = 0 (padding, punctuation) are zero."""
        x = self.embeddings(ids)
        for block in self.blocks:
            x, _ = block(x, mask)
        v = self.linear(x)
        if keep is not None:
            v = v * keep[..., None]
        return F.normalize(v, dim=-1)


# %% [maxsim] MaxSim: late interaction
def maxsim(q, d, d_keep):
    """q: (32, 128) query vectors; d: (n, 128) paragraph vectors; d_keep: (n,) which paragraph tokens count.
    Returns the (32, n) cosine matrix, each query token's best paragraph token, and the score = the sum of the maxima."""
    sim = q @ d.T
    sim = sim.masked_fill(~d_keep[None, :], -9999.0)  # dropped tokens can never be a maximum
    best, where = sim.max(dim=-1)
    return sim, where, best.sum()


# %% [rank] Scoring the whole corpus
def score_corpus(q, docs, keep):
    """q: (queries, 32, 128); docs: (paragraphs, longest, 128) zero-padded; keep: (paragraphs, longest). Exhaustive
    MaxSim against every paragraph (no index): (queries, paragraphs) scores."""
    sim = torch.einsum("qid,pjd->qpij", q, docs)
    sim = sim.masked_fill(~keep[None, :, None, :], -9999.0)
    return sim.max(-1).values.sum(-1)
`;export{n as default};
