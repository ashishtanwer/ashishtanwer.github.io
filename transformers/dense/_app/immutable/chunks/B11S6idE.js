const e=`# HNSW (Malkov & Yashunin 2016) over SQuAD's 2,067 paragraphs, the HNSW tab, as plain PyTorch.
# Every paragraph is a node; its bge-small vector (384 numbers, unit length) is its position. Each node gets a random
# top layer: all nodes are on layer 0, about 1 in M also on layer 1, 1 in M² on layer 2 … Each layer is a graph that
# links a node to up to M nearby nodes (2M on layer 0). A query starts at the top layer's entry point, walks greedily
# to its nearest node on each upper layer, then on layer 0 keeps a list of the ef nearest nodes found so far and
# expands them until every one has been expanded. The best k in that list are the answer.
# This file follows faiss 1.15 (IndexHNSWFlat) step for step; the tab's graph is faiss's, its search trace is this
# file's search(), checked to return what faiss returns. Each "# %% [id]" section is one card of the tab.
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    M: int = 16  # links per node on layers ≥ 1; layer 0 allows 2M = 32
    ef_construction: int = 40  # candidate-list size while inserting a node
    ef_search: int = 16  # candidate-list size while searching (faiss's default)
    k: int = 10  # results returned
    prune_headroom: float = 0.2  # a full list that must take a new link is pruned to 80 % of its slots
    dim: int = 384  # bge-small-en-v1.5


# %% [distance] Distance
def distances(query, vectors):
    """Squared Euclidean distance from one query (dim,) to vectors (n, dim). For unit vectors it equals 2 − 2·cos,
    so the nearest node by distance is the most similar by cosine; faiss's IndexHNSWFlat uses it (METRIC_L2)."""
    return ((vectors - query) ** 2).sum(-1)


# %% [levels] Layers
def level_probabilities(M):
    """P(top layer = l) = e^(−l/m)·(1 − e^(−1/m)) with m = 1/ln M, i.e. (1 − 1/M)·M^−l: each layer keeps about 1/M of
    the nodes of the layer below. faiss stops the table where the probability drops under 1e-9."""
    m = 1 / math.log(M)
    probs, level = [], 0
    while (p := math.exp(-level / m) * (1 - math.exp(-1 / m))) >= 1e-9:
        probs.append(p)
        level += 1
    return probs


def draw_levels(uniforms, M):
    """Each node's top layer from one uniform number in [0, 1]: walk down the probability table until the number
    falls inside a level's slice. uniforms: (n,) float64. Returns (n,) top layers (0 = only on the bottom layer)."""
    probs = level_probabilities(M)
    levels = torch.full(uniforms.shape, len(probs) - 1, dtype=torch.long)
    for i, f in enumerate(uniforms.tolist()):
        for level, p in enumerate(probs):
            if f < p:
                levels[i] = level
                break
            f -= p
    return levels


# %% [greedy] Greedy descent on an upper layer
def greedy(query, vectors, links, nearest, d_nearest):
    """Move to whichever neighbour of the current node is closest to the query, until none is closer.
    links: (n, M) neighbour ids of this layer, −1 where a slot is empty. Returns the node reached, its distance,
    the hops [(node, distance)] and how many distances were computed."""
    hops, computed = [(nearest, d_nearest)], 0
    while True:
        here = nearest
        around = links[here]
        around = around[around >= 0]
        d = distances(query, vectors[around])
        computed += len(around)
        for node, dist in zip(around.tolist(), d.tolist()):  # in list order; only a strictly closer one replaces
            if dist < d_nearest:
                nearest, d_nearest = node, dist
        if nearest == here:
            return nearest, d_nearest, hops, computed
        hops.append((nearest, d_nearest))


# %% [search-layer] Best-first search on layer 0
def search_layer0(query, vectors, links, entry, d_entry, ef, k):
    """Keep a candidate list of the ef nearest nodes found so far, each marked expanded or not. Repeatedly expand the
    nearest unexpanded one: compute the distance to each of its neighbours not seen before, and let a neighbour into
    the list if the list has room or it beats the list's farthest entry (which it then replaces). Stop when every
    node in the list is expanded. The answer is the k nearest of all nodes whose distance was computed.
    links: (n, 2M) layer-0 neighbour ids. Returns (ids, distances) of the k results, sorted, and a trace."""
    size = max(ef, k)  # the list never holds fewer slots than results asked for
    seen = {entry}
    found = {entry: d_entry}  # every node whose distance was computed (faiss keeps the best k in a heap)
    candidates = [[d_entry, entry, False]]  # [distance, node, expanded]
    trace = []
    while any(not c[2] for c in candidates):
        current = min((c for c in candidates if not c[2]), key=lambda c: c[0])
        current[2] = True
        if sum(c[0] < current[0] for c in candidates) >= ef:  # only possible when ef < k: ef better nodes in hand
            break
        around = links[current[1]]
        around = [n for n in around[around >= 0].tolist() if n not in seen]
        seen.update(around)
        d = distances(query, vectors[around]).tolist() if around else []
        step = {'node': current[1], 'distance': current[0], 'evaluated': [], 'admitted': []}
        for node, dist in zip(around, d):
            found[node] = dist
            farthest = max(candidates, key=lambda c: c[0])
            admitted = len(candidates) < size or dist < farthest[0]
            if admitted:
                if len(candidates) == size:
                    candidates.remove(farthest)  # the farthest goes, expanded or not
                candidates.append([dist, node, False])
            step['evaluated'].append((node, dist))
            step['admitted'].append(admitted)
        trace.append(step)
    best = sorted(found.items(), key=lambda item: item[1])[:k]
    return [node for node, _ in best], [dist for _, dist in best], trace


# %% [search] One query
def search(query, vectors, links, entry_point, ef, k):
    """Greedy descent from the entry point through layers top … 1 (one node kept per layer), then the best-first
    search on layer 0 with a list of max(ef, k). links: list of (n, width) neighbour tables, one per layer.
    Returns result ids, their distances, the path and the number of distance computations."""
    nearest = entry_point
    d_nearest = distances(query, vectors[nearest : nearest + 1]).item()
    computed, path = 1, []
    for level in range(len(links) - 1, 0, -1):
        nearest, d_nearest, hops, n = greedy(query, vectors, links[level], nearest, d_nearest)
        computed += n
        path.append({'level': level, 'hops': hops, 'computed': n})
    ids, dists, trace = search_layer0(query, vectors, links[0], nearest, d_nearest, ef, k)
    computed += sum(len(step['evaluated']) for step in trace)
    return ids, dists, {'upper': path, 'layer0': trace}, computed


# %% [recall] Recall against exact search
def exact_search(queries, vectors, k):
    """Every distance, the k smallest: what an index approximates. queries (q, dim), vectors (n, dim)."""
    d = torch.cdist(queries.double(), vectors.double()) ** 2
    return d.topk(k, largest=False).indices


def recall_at_k(found, exact):
    """Fraction of the exact top-k that the index returned, averaged over the queries. found, exact: (q, k) ids."""
    hits = [len(set(f.tolist()) & set(e.tolist())) for f, e in zip(found, exact)]
    return torch.tensor(hits, dtype=torch.float64) / exact.shape[1]


# %% [select] Choosing a node's neighbours
def select_neighbors(vectors, candidates, max_size):
    """faiss's heuristic (the paper's Algorithm 4): go through the candidates nearest first and keep one only if it
    is closer to the node than to every neighbour already kept, so the links spread out in different directions
    instead of all pointing into one cluster. candidates: [(distance to the node, id)], max_size: links allowed.
    With fewer candidates than max_size, faiss keeps them all."""
    if len(candidates) < max_size:
        return sorted(candidates)
    kept = []
    for dist, node in sorted(candidates):
        others = torch.tensor([other for _, other in kept], dtype=torch.long)
        if not kept or bool((distances(vectors[node], vectors[others]) >= dist).all()):
            kept.append((dist, node))
            if len(kept) >= max_size:
                break
    return kept


# %% [build] Building the graph
def search_to_insert(query, vectors, links, entry, d_entry, ef):
    """The search used while inserting: best-first from one entry node, keeping the ef nearest found; stop when the
    nearest unexpanded candidate is farther than the farthest kept. Returns [(distance, id)] of the kept nodes."""
    kept, frontier, seen = [(d_entry, entry)], [(d_entry, entry)], {entry}
    while frontier:
        frontier.sort()
        d, node = frontier[0]
        if d > max(kept)[0]:
            break
        frontier.pop(0)
        around = [n for n in links[node][links[node] >= 0].tolist() if n not in seen]
        seen.update(around)
        for n, dn in zip(around, distances(query, vectors[around]).tolist() if around else []):
            if len(kept) < ef or dn < max(kept)[0]:
                kept.append((dn, n))
                frontier.append((dn, n))
                if len(kept) > ef:
                    kept.remove(max(kept))
    return kept


def build(vectors, levels, shuffle, cfg=Config(), log=None):
    """faiss 1.15's deterministic build: nodes are inserted from the top layer's down, in a shuffled order within each
    layer's group, in batches of 1, 1, 2, 4 … (at most 2 % of the corpus). Each node of a batch searches the graph as it
    was before the batch (greedy down to its top layer, then a search of ef_construction on each of its layers),
    keeps M (2M on layer 0) neighbours with select_neighbors, and writes them, farthest first. Then every chosen
    neighbour takes a link back; a list that overflows keeps its nearest by select_neighbors, pruned to 80 %.
    levels: (n,) top layers from draw_levels; shuffle: iterator of 32-bit random integers (faiss's mt19937 seeded
    with 789). Returns (links per layer as (n, width) tensors, −1 for empty slots; the entry point). A dict passed as
    log receives each node's insertion: batch, and per layer the candidates found, those kept, the links back."""
    n, top = len(vectors), int(levels.max())
    widths = [2 * cfg.M] + [cfg.M] * top
    links = [torch.full((n, w), -1, dtype=torch.long) for w in widths]
    entry, max_level, batch_no = -1, -1, 0
    for level in range(top, -1, -1):
        group = [i for i in range(n) if levels[i] == level]
        for j in range(len(group)):  # faiss's in-place shuffle of the group
            swap = j + next(shuffle) % (len(group) - j)
            group[j], group[swap] = group[swap], group[j]
        if entry == -1:
            entry, max_level = group[0], level
        start, theta = 0, max(1, int(0.02 * n))
        while start < len(group):
            batch_no += 1
            batch = group[start : start + min(start or 1, theta)]
            reverse = {}  # (layer, neighbour) → nodes of this batch that linked to it
            for node in batch:  # each sees only the graph from before the batch
                d_near = distances(vectors[node], vectors[entry : entry + 1]).item()
                nearest = entry
                for layer in range(max_level, level, -1):
                    nearest, d_near, _, _ = greedy(vectors[node], vectors, links[layer], nearest, d_near)
                for layer in range(min(level, max_level), -1, -1):
                    found = search_to_insert(vectors[node], vectors, links[layer], nearest, d_near, cfg.ef_construction)
                    chosen = [i for _, i in sorted(select_neighbors(vectors, found, widths[layer]), reverse=True)]
                    chosen = [i for i in chosen if i != node]
                    links[layer][node, : len(chosen)] = torch.tensor(chosen, dtype=torch.long)
                    if log is not None:
                        log.setdefault(node, {'batch': batch_no})[layer] = {'found': found, 'kept': chosen, 'back': []}
                    for other in chosen:
                        reverse.setdefault((layer, other), []).append(node)
            for (layer, other), incoming in reverse.items():
                row = links[layer][other]
                current = row[row >= 0].tolist()
                current += [i for i in sorted(set(incoming)) if i != other and i not in current]
                if log is not None:
                    log[other][layer]['back'].append((sorted(set(incoming)), len(current) > widths[layer]))
                if len(current) > widths[layer]:
                    d = distances(vectors[other], vectors[current]).tolist()
                    pruned = int(widths[layer] - widths[layer] * cfg.prune_headroom)
                    current = [i for _, i in select_neighbors(vectors, sorted(zip(d, current)), pruned)]
                row.fill_(-1)
                row[: len(current)] = torch.tensor(current, dtype=torch.long)
            start += len(batch)
    return links, entry
`;export{e as default};
