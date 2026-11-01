const e=`# Hybrid search: the Retrieval Explainer's Hybrid tab as plain PyTorch. Two retrievers rank the same corpus, BM25 by
# exact term matches (the BM25 tab) and BAAI/bge-small-en-v1.5 by the cosine of sentence embeddings (the Embeddings
# tab); each keeps its top 100. Reciprocal rank fusion (Cormack et al., 2009) adds 1 / (k + rank) from every list a
# paragraph appears in, so it needs ranks only, never the two methods' incomparable scores. The alternative rescales
# each list's scores to 0–1 (min–max) and takes a weighted sum (a convex combination, Bruch et al., 2023).
from dataclasses import dataclass

import torch


@dataclass
class Config:
    depth: int = 100  # paragraphs each retriever contributes
    k: float = 60.0  # RRF's constant: larger k flattens the reward for being near the top
    alpha: float = 0.5  # the dense list's weight in the convex combination


# %% [lists] Two ranked lists
def top(scores, depth: int):
    """Each query's \`depth\` best paragraphs: (ids, scores), best first. A stable sort, so ties keep corpus order."""
    values, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    return order[:, :depth], values[:, :depth]


def dense_scores(queries, docs):
    """bge-small's unit vectors: the cosine is the dot product. (queries, docs)."""
    return queries @ docs.T


# %% [rrf] Reciprocal rank fusion
def rrf(lists, docs: int, k: float):
    """lists: [(queries, depth) ids, best first] from each retriever. Every paragraph gets Σ 1 / (k + rank) over the
    lists that hold it (rank from 1); a paragraph a list does not hold gets nothing from it. (queries, docs)."""
    fused = torch.zeros(lists[0].shape[0], docs, dtype=torch.float64)
    for ids in lists:
        rank = torch.arange(1, ids.shape[1] + 1, dtype=torch.float64).expand(ids.shape)
        fused.scatter_add_(1, ids, 1.0 / (k + rank))
    return fused


# %% [convex] Min–max normalization and a weighted sum
def minmax(ids, scores, docs: int):
    """Each list's scores rescaled to 0 (its last) … 1 (its first), scattered over the corpus; paragraphs outside the
    list get 0. (queries, docs)."""
    low, high = scores.min(-1, keepdim=True).values, scores.max(-1, keepdim=True).values
    out = torch.zeros(ids.shape[0], docs, dtype=torch.float64)
    return out.scatter_(1, ids, ((scores - low) / (high - low)).double())


def convex(sparse, dense, alpha: float):
    """α · dense + (1 − α) · sparse on the normalized scores: α = 0 is BM25's order, α = 1 bge-small's."""
    return alpha * dense + (1 - alpha) * sparse


# %% [metrics] Fused ranking, recall and nDCG
def fused_rank(fused, depth: int):
    """The fused order, best first (ties keep corpus order): (queries, depth) ids."""
    return torch.sort(fused, dim=-1, descending=True, stable=True)[1][:, :depth]


def metrics(ranked, relevant, ks=(1, 5, 10, 100)):
    """ranked: (queries, depth) paragraph ids, best first; relevant: (queries,) the one relevant id per query."""
    hit = ranked == relevant[:, None]
    found = hit.any(-1)
    rank1 = hit.float().argmax(-1) + 1
    out = {f"recall@{k}": (hit[:, :k].any(-1)).float().mean() for k in ks}
    out["mrr@10"] = torch.where(found & (rank1 <= 10), 1.0 / rank1, 0.0).mean()
    out["ndcg@10"] = torch.where(found & (rank1 <= 10), 1.0 / torch.log2(rank1 + 1.0), 0.0).mean()
    return out
`;export{e as default};
