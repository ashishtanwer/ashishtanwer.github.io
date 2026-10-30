const e=`# HiCache, SGLang's hierarchical KV cache, for Qwen2.5-0.5B-Instruct, the model in the HiCache tab, as plain
# PyTorch. A radix tree of cached pages records which tier holds each node: the GPU pool, the host pool
# (CPU memory) or storage (one file per page). Requests reuse their longest cached prefix from any tier.
# The model is any causal LM with a key/value cache (the KV-cache tab's). Each "# %% [id]" section is one block.
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    page: int = 16  # tokens per page; only whole pages are cached
    gpu_pages: int = 14
    host_pages: int = 28  # more than the GPU's (SGLang requires it)
    prefetch_threshold: int = 32  # tokens: shorter storage hits are recomputed
    selective_hits: int = 2
    policy: str = "write_through"  # or "write_through_selective", "write_back"
    layers: int = 24
    kv_heads: int = 2
    head_dim: int = 64


# %% [radix-tree] Radix tree of pages
@dataclass
class Node:
    id: int
    parent: "Node | None"
    pages: list  # page ids, in order
    children: dict = field(default_factory=dict)  # first page's tokens → Node
    gpu: list | None = None  # GPU slot of each page
    host: list | None = None  # host slot of each page
    storage: bool = False  # every page has its file
    last: int = 0
    hits: int = 0
    locked: bool = False


class RadixTree:
    def __init__(self):
        self.root = Node(0, None, [])
        self.nodes = {0: self.root}
        self.tokens = {}  # page id → its tokens

    def split(self, node, k):
        """Split after the first k pages, so a match can end inside what was one node."""
        head = Node(max(self.nodes) + 1, node.parent, node.pages[:k], last=node.last, hits=node.hits, storage=node.storage)
        head.gpu = node.gpu[:k] if node.gpu is not None else None
        head.host = node.host[:k] if node.host is not None else None
        node.parent.children[self.tokens[node.pages[0]]] = head
        node.pages, node.parent = node.pages[k:], head
        node.gpu = node.gpu[k:] if node.gpu is not None else None
        node.host = node.host[k:] if node.host is not None else None
        head.children[self.tokens[node.pages[0]]] = node
        self.nodes[head.id] = head
        return head

    def match(self, pages):
        """The nodes along the longest cached prefix of \`pages\` (each a tuple of tokens)."""
        path, node, i = [], self.root, 0
        while i < len(pages) and pages[i] in node.children:
            child, k = node.children[pages[i]], 0
            while k < len(child.pages) and i + k < len(pages) and self.tokens[child.pages[k]] == pages[i + k]:
                k += 1
            if k < len(child.pages):
                child = self.split(child, k)
            path.append(child)
            node, i = child, i + k
        return path


# %% [tiers] Three tiers, write policies, eviction
class HiCache(RadixTree):
    """GPU pool (layer-first), host pool (page-first), storage files. Writes follow the policy:
    write_through copies a new node down at once; write_through_selective once it has been hit
    selective_hits times; write_back only when it is evicted from the tier above. Eviction takes the
    least recently used leaf; an evicted GPU node with no host copy is dropped (or, under write_back,
    first written back)."""

    def __init__(self, c: Config, folder: Path, device="cpu"):
        super().__init__()
        shape = (c.kv_heads, c.page, c.head_dim)
        self.c, self.folder, self.device = c, Path(folder), device
        self.gpu = torch.zeros(c.layers, 2, c.gpu_pages, *shape, device=device)
        self.host = torch.zeros(c.host_pages, c.layers, 2, *shape)
        self.free_gpu, self.free_host = list(range(c.gpu_pages)), list(range(c.host_pages))
        self.page_bytes = self.host[0].numel() * self.host.element_size()
        self.moved = {}  # kind → bytes
        self.clock = 0

    def _move(self, kind, node):
        self.moved[kind] = self.moved.get(kind, 0) + len(node.pages) * self.page_bytes

    def gpu_to_host(self, node):
        node.host = self._slots("host", len(node.pages), node)
        self.host[node.host] = self.gpu[:, :, node.gpu].permute(2, 0, 1, 3, 4, 5).cpu()
        self._move("gpu→host", node)
        if self.c.policy != "write_back" and not node.storage:
            self.host_to_storage(node)

    def host_to_storage(self, node):
        for page, slot in zip(node.pages, node.host):
            (self.folder / f"{page}.kv").write_bytes(self.host[slot].numpy().tobytes())
        node.storage = True
        self._move("host→storage", node)

    def storage_to_host(self, node):
        node.host = self._slots("host", len(node.pages), node)
        for page, slot in zip(node.pages, node.host):
            data = bytearray((self.folder / f"{page}.kv").read_bytes())
            self.host[slot] = torch.frombuffer(data, dtype=torch.float32).view(self.host.shape[1:])
        self._move("storage→host", node)

    def host_to_gpu(self, node):
        node.gpu = self._slots("gpu", len(node.pages), node)
        self.gpu[:, :, node.gpu] = self.host[node.host].permute(1, 2, 0, 3, 4, 5).to(self.device)
        self._move("host→gpu", node)

    def _slots(self, tier, n, keep):
        free = self.free_gpu if tier == "gpu" else self.free_host
        while len(free) < n:
            self.evict(tier, keep)
        taken = free[:n]
        del free[:n]
        return taken

    def _leaf(self, x, tier):
        on = (lambda y: y.gpu is not None) if tier == "gpu" else (lambda y: y.host is not None and y.gpu is None)
        return x is not self.root and on(x) and not x.locked and not any(on(c) for c in x.children.values())

    def evict(self, tier, keep):
        node = min((x for x in self.nodes.values() if x is not keep and self._leaf(x, tier)), key=lambda x: (x.last, x.id))
        if tier == "gpu":
            if node.host is None and self.c.policy == "write_back":
                self.gpu_to_host(node)
            self.free_gpu += node.gpu
            node.gpu = None
        else:
            if not node.storage and self.c.policy == "write_back":
                self.host_to_storage(node)
            self.free_host += node.host
            node.host = None
        if node.gpu is None and node.host is None and not node.storage:  # nowhere: it and its subtree go
            stack = [node]
            while stack:
                x = stack.pop()
                stack += list(x.children.values())
                self.free_gpu += x.gpu or []
                self.free_host += x.host or []
                self.nodes.pop(x.id, None)
            del node.parent.children[self.tokens[node.pages[0]]]

    def insert(self, pages, kv, parent):
        """New pages (tokens, and their KV as (layers, 2, kv_heads, page, head_dim)) as one node on the GPU."""
        ids = []
        for tokens in pages:
            self.tokens[len(self.tokens)] = tokens
            ids.append(len(self.tokens) - 1)
        node = Node(max(self.nodes) + 1, parent, ids, last=self.clock, locked=True)
        parent.children[pages[0]] = node
        self.nodes[node.id] = node
        node.gpu = self._slots("gpu", len(ids), node)
        self.gpu[:, :, node.gpu] = torch.stack(kv).permute(1, 2, 0, 3, 4, 5).to(self.device)
        self._move("write gpu", node)
        if self.c.policy == "write_through":
            self.gpu_to_host(node)
        node.locked = False
        return node


# %% [prefetch] Prefetch and load-back
def load(cache: HiCache, path):
    """Before the prefill: pages only in storage are prefetched into host memory (if the storage hit is
    at least prefetch_threshold tokens; otherwise the match stops there and they are recomputed), then
    every page not on the GPU is copied there. Returns the nodes the prefill reads, in order."""
    c = cache.c
    tiers = ["gpu" if n.gpu is not None else "host" if n.host is not None else "storage" for n in path]
    stored = sum(len(n.pages) for n, t in zip(path, tiers) if t == "storage") * c.page
    if 0 < stored < c.prefetch_threshold:
        cut = tiers.index("storage")
        for n in path[cut:]:
            n.locked = False
        path, tiers = path[:cut], tiers[:cut]
    for node, tier in zip(path, tiers):
        if tier == "storage":
            cache.storage_to_host(node)
        if node.gpu is None:
            cache.host_to_gpu(node)
        node.hits, node.last = node.hits + 1, cache.clock
        hot = c.policy == "write_through" or (c.policy == "write_through_selective" and node.hits >= c.selective_hits)
        if node.host is None and hot:
            cache.gpu_to_host(node)
    return path, tiers


# %% [serve] Serving a request
@torch.no_grad()
def serve(model, cache: HiCache, ids, max_new_tokens=32, eos=151645, answer=None):
    """Match, load, prefill only the uncached part of the prompt, decode, insert prompt + answer.
    model(ids, past) → (logits of the last position, past) with past a list of (k, v) per layer,
    each (kv_heads, seq, head_dim). \`answer\` feeds a known answer instead of decoding."""
    c = cache.c
    cache.clock += 1
    pages = [tuple(ids[p * c.page : (p + 1) * c.page].tolist()) for p in range(len(ids) // c.page)]
    path = cache.match(pages)
    for node in path:
        node.locked = True
    path, tiers = load(cache, path)
    cached = sum(len(n.pages) for n in path) * c.page
    slots = [s for n in path for s in n.gpu]
    past = [(cache.gpu[l, 0, slots].transpose(0, 1).flatten(1, 2), cache.gpu[l, 1, slots].transpose(0, 1).flatten(1, 2)) for l in range(c.layers)]
    logits, past = model(ids[cached:], past)
    first, tokens = logits, []
    while len(tokens) < max_new_tokens:
        token = int(logits.argmax()) if answer is None else answer[len(tokens)]
        tokens.append(token)
        if token == eos or len(tokens) == max_new_tokens or (answer is not None and len(tokens) == len(answer)):
            break
        logits, past = model(torch.tensor([token]), past)
    sequence = torch.cat((ids, torch.tensor(tokens[:-1])))  # the last token was never fed
    whole = len(sequence) // c.page
    all_pages = [tuple(sequence[p * c.page : (p + 1) * c.page].tolist()) for p in range(whole)]
    known = cache.match(all_pages)
    start = sum(len(n.pages) for n in known)
    if start < whole:
        kv = [torch.stack([torch.stack((k[:, p * c.page : (p + 1) * c.page], v[:, p * c.page : (p + 1) * c.page])) for k, v in past]).cpu() for p in range(start, whole)]
        cache.insert(all_pages[start:], kv, known[-1] if known else cache.root)
    for node in path:
        node.locked = False
    return tokens, first, cached, tiers


# cache = HiCache(Config(policy="write_back"), folder); for ids in requests: serve(qwen, cache, ids)
`;export{e as default};
