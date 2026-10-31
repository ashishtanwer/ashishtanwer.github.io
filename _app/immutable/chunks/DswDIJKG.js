const e=`# IVF-PQ (Jégou, Douze & Schmid 2011) over SQuAD's 2,067 paragraphs, the IVF-PQ tab, as plain PyTorch.
# Two quantizers. The coarse one is k-means with nlist centroids: every vector is filed under its nearest centroid
# (an inverted list, or cell), and a query only scans the nprobe cells whose centroids are nearest to it. The product
# quantizer stores each vector as m bytes: its residual from the cell's centroid is cut into m sub-vectors, and each
# is replaced by the index of its nearest of 256 sub-centroids (a codebook per sub-space, also k-means). A query is
# not quantized: per probed cell it computes a table of its distance to every sub-centroid, and a vector's distance is
# then m table lookups added up (asymmetric distance computation).
# This file follows faiss 1.15 (IndexIVFPQ, by_residual, 8-bit codes) step for step; the tab's index is faiss's and
# this file reproduces its centroids, codebooks, codes and search results. Each "# %% [id]" section is one card.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    dim: int = 384  # bge-small-en-v1.5
    nlist: int = 64  # coarse cells
    m: int = 16  # sub-vectors (bytes per code)
    nbits: int = 8  # 2^8 = 256 sub-centroids per sub-space
    coarse_iters: int = 10  # faiss's IVF k-means
    pq_iters: int = 25  # faiss's PQ k-means, per sub-space
    seed: int = 1234  # faiss's ClusteringParameters seed; the first centroids are a shuffle seeded with seed + 1
    nprobe: int = 1  # cells scanned per query (faiss's default)
    k: int = 10


# %% [kmeans] k-means
def random_order(n, draws):
    """faiss's rand_perm: a Fisher–Yates shuffle of 0 … n−1 driven by 32-bit random integers (its mt19937)."""
    order = list(range(n))
    for i in range(n - 1):
        j = i + next(draws) % (n - i)
        order[i], order[j] = order[j], order[i]
    return order


def kmeans(x, k, iters, first):
    """Lloyd's algorithm from k starting points: give every vector to its nearest centroid, move each centroid to
    the mean of its vectors, repeat. x: (n, d); first: the n indices in a random order, whose first k are the start.
    Returns the centroids (k, d) and the final assignment (n,). (faiss would split a big cell to refill an empty
    one; none came up empty on this corpus.)"""
    x = x.double()
    centroids = x[first[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(1)
        counts = torch.bincount(assign, minlength=k)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        centroids = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centroids)
    return centroids, torch.cdist(x, centroids).argmin(1)


# %% [assign] Inverted lists
def assign(x, centroids):
    """Each vector's cell: the index of its nearest coarse centroid. The cells' member lists are the inverted lists."""
    return torch.cdist(x.double(), centroids.double()).argmin(1)


# %% [pq-train] Product-quantizer codebooks
def train_codebooks(residuals, cfg, first):
    """One k-means of 2^nbits centroids per sub-space, on that sub-space's slice of the training residuals.
    residuals: (n, dim) → codebooks (m, 256, dim / m)."""
    sub = cfg.dim // cfg.m
    return torch.stack(
        [kmeans(residuals[:, j * sub : (j + 1) * sub], 2**cfg.nbits, cfg.pq_iters, first)[0] for j in range(cfg.m)]
    )


# %% [encode] Encoding a vector in m bytes
def encode(x, centroids, cells, codebooks):
    """residual = x − its cell's centroid, cut into m sub-vectors; each becomes the index of its nearest
    sub-centroid. x: (n, dim), cells: (n,) → codes (n, m), integers 0 … 255."""
    m, ksub, sub = codebooks.shape
    residual = x.double() - centroids.double()[cells]
    return torch.stack(
        [torch.cdist(residual[:, j * sub : (j + 1) * sub], codebooks[j].double()).argmin(1) for j in range(m)], 1
    )


def decode(codes, cells, centroids, codebooks):
    """What the index remembers of a vector: its cell's centroid plus the m chosen sub-centroids, side by side."""
    m = codebooks.shape[0]
    parts = torch.cat([codebooks[j].double()[codes[:, j]] for j in range(m)], 1)  # (n, dim)
    return centroids.double()[cells] + parts


# %% [adc] Asymmetric distance table
def distance_table(query, centroid, codebooks):
    """For one probed cell: the squared distance from each sub-vector of the query's residual (query − centroid) to
    each of the 256 sub-centroids of its sub-space. (m, 256). The query itself is never quantized."""
    m, ksub, sub = codebooks.shape
    r = query.double() - centroid.double()
    return torch.stack([((codebooks[j].double() - r[j * sub : (j + 1) * sub]) ** 2).sum(-1) for j in range(m)])


def adc(table, codes):
    """A code's estimated squared distance: one table lookup per byte, added up. codes: (n, m) → (n,)."""
    return table[torch.arange(table.shape[0]), codes].sum(-1)


# %% [search] Probing nprobe cells
def search(query, centroids, cells, codes, codebooks, nprobe, k):
    """Rank the cells by the distance from the query to their centroids, scan the codes of the nprobe nearest with
    their tables, keep the k smallest estimates. Returns ids, estimated distances, the probed cells and how many
    codes were scanned."""
    to_cells = ((centroids.double() - query.double()) ** 2).sum(-1)  # (nlist,)
    probed = to_cells.argsort()[:nprobe]
    ids, dists = [], []
    for cell in probed.tolist():
        members = torch.nonzero(cells == cell)[:, 0]
        table = distance_table(query, centroids[cell], codebooks)
        ids.append(members)
        dists.append(adc(table, codes[members]))
    ids, dists = torch.cat(ids), torch.cat(dists)
    best = dists.argsort(stable=True)[:k]
    return ids[best], dists[best], probed, len(ids)


# %% [rerank] Re-ranking with the full vectors
def rerank(query, vectors, ids, k):
    """faiss's IndexRefineFlat: search the codes for k · k_factor candidates, then compute their exact distances from
    the stored float vectors and keep the k nearest. Recall comes back, but the float vectors must be kept (in memory
    or on disk) after all. ids: (k · k_factor,) candidates from search()."""
    d = ((vectors[ids].double() - query.double()) ** 2).sum(-1)
    order = d.argsort(stable=True)[:k]
    return ids[order], d[order]


# %% [recall] Recall against exact search
def exact_search(queries, vectors, k):
    """Every distance, the k smallest: what an index approximates. queries (q, dim), vectors (n, dim)."""
    d = torch.cdist(queries.double(), vectors.double()) ** 2
    return d.topk(k, largest=False).indices


def recall_at_k(found, exact):
    """Fraction of the exact top-k that the index returned, per query. found, exact: (q, k) ids."""
    hits = [len(set(f.tolist()) & set(e.tolist())) for f, e in zip(found, exact)]
    return torch.tensor(hits, dtype=torch.float64) / exact.shape[1]


# %% [bytes] Memory per vector
def bytes_per_vector(cfg, id_bytes=8):
    """faiss stores, per vector, its m-byte code and its 8-byte id in the cell's list; the float32 vector would be
    4 · dim bytes. The centroids and codebooks are shared: nlist · dim + 256 · dim floats in all."""
    return {
        'float32': 4 * cfg.dim,
        'code': cfg.m * cfg.nbits // 8,
        'id': id_bytes,
        'shared': 4 * (cfg.nlist * cfg.dim + 2**cfg.nbits * cfg.dim),
    }
`;export{e as default};
