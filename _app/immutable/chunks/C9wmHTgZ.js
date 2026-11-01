const e=`# Continuous batching, the method in the Efficiency Explainer's Batching tab, written as plain PyTorch blocks.
# A server keeps a running batch and rebuilds it at every step: each running request contributes its next token, a
# finished request leaves at once and frees its KV blocks, and a waiting request joins as soon as a seat and blocks are
# free (iteration-level scheduling, Orca; paged KV blocks, vLLM). Static batching starts a new batch only when every
# request of the last one has finished. Shown on Qwen2.5-0.5B-Instruct; the linear layers run once over every token
# of the step, attention reads each request's own blocks. Each "# %% [id]" section is one block of the page.
import math
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Qwen2.5-0.5B-Instruct, served
    vocab_size: int = 151936
    d_model: int = 896
    n_layers: int = 24
    n_heads: int = 14  # 64 dimensions per head
    n_kv_heads: int = 2  # grouped-query attention: seven query heads share each key/value head
    d_mlp: int = 4864
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    block_size: int = 16  # positions per KV block (vLLM's default)
    num_blocks: int = 96  # the KV pool every request allocates from
    max_num_seqs: int = 4  # requests running at once
    max_num_batched_tokens: int = 2048  # tokens one step may run
    eos: int = 151645  # <|im_end|>

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


# %% [rope] RoPE at the tokens' positions
def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d). Every token of a step is
    rotated by its own position in its own request. x: (tokens, heads, d). positions: (tokens,)."""
    d = x.shape[-1]
    frequencies = (1.0 / theta ** (torch.arange(0, d, 2).float() / d)).to(x.device)  # computed on the CPU
    angles = positions[:, None].float() * frequencies  # (tokens, d/2)
    cos, sin = angles.cos().repeat(1, 2)[:, None], angles.sin().repeat(1, 2)[:, None]
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [paged] One pool of KV blocks for every request
class BlockPool:
    """vLLM's paged layout: per layer, K and V pools of (num_blocks, block_size, kv_heads, head_dim). A request holds a
    block table (its logical blocks' physical ids), takes a block from the free list only when its last one fills,
    and gives all of them back when it finishes, to the end of the free list, last block first, as vLLM's free queue
    takes them."""

    def __init__(self, c: Config, device="cpu"):
        shape = (c.n_layers, c.num_blocks, c.block_size, c.n_kv_heads, c.head_dim)
        self.c, self.k, self.v = c, torch.zeros(shape, device=device), torch.zeros(shape, device=device)
        self.free = list(range(c.num_blocks))

    def needed(self, table, length):
        return max(0, math.ceil(length / self.c.block_size) - len(table))

    def allocate(self, table, length):
        """Grow a block table to hold \`length\` positions; False (and no change) if the pool is short."""
        if self.needed(table, length) > len(self.free):
            return False
        while len(table) * self.c.block_size < length:
            table.append(self.free.pop(0))
        return True

    def release(self, table):
        self.free.extend(reversed(table))
        table.clear()

    def slots(self, table, start, end):
        """Flat slot index (block × block_size + offset) of positions start … end − 1."""
        size = self.c.block_size
        return torch.tensor([table[p // size] * size + p % size for p in range(start, end)])


# %% [attention] Attention over each request's blocks
class Attention(nn.Module):
    """The step's tokens from every request are projected together; each token's key and value are written to its
    request's slots; then each request's queries read that request's blocks only. Decodes (one query each) are
    gathered into one padded batch, as a paged-attention kernel reads them; a prefill reads its own blocks."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def forward(self, x, batch, pool, layer):  # x: (tokens, d_model), every request's new tokens one after another
        c, t, group = self.c, x.shape[0], self.c.n_heads // self.c.n_kv_heads
        q = rope(self.q(x).view(t, c.n_heads, c.head_dim), batch.positions, c.rope_theta)
        k = rope(self.k(x).view(t, c.n_kv_heads, c.head_dim), batch.positions, c.rope_theta)
        v = self.v(x).view(t, c.n_kv_heads, c.head_dim)
        pool.k[layer].view(-1, c.n_kv_heads, c.head_dim)[batch.slots] = k  # write this step's rows
        pool.v[layer].view(-1, c.n_kv_heads, c.head_dim)[batch.slots] = v
        out = torch.empty_like(q)
        if batch.decode_rows is not None:  # (requests, blocks × block_size, kv_heads, head_dim), padded; padding masked
            K = pool.k[layer][batch.decode_tables].flatten(1, 2).repeat_interleave(group, dim=2)
            V = pool.v[layer][batch.decode_tables].flatten(1, 2).repeat_interleave(group, dim=2)
            scores = torch.einsum("bhd,blhd->bhl", q[batch.decode_rows], K) / math.sqrt(c.head_dim)
            visible = torch.arange(K.shape[1], device=x.device)[None, :] < batch.decode_lengths[:, None]
            weights = scores.masked_fill(~visible[:, None, :], float("-inf")).softmax(dim=-1)
            out[batch.decode_rows] = torch.einsum("bhl,blhd->bhd", weights, V)
        for mine, blocks, length in batch.prefills:  # causal attention over the request's own blocks
            K = pool.k[layer][blocks].flatten(0, 1)[:length].repeat_interleave(group, dim=1)  # (length, heads, d)
            V = pool.v[layer][blocks].flatten(0, 1)[:length].repeat_interleave(group, dim=1)
            scores = torch.einsum("thd,lhd->htl", q[mine], K) / math.sqrt(c.head_dim)
            visible = torch.arange(length, device=x.device)[None, :] <= batch.positions[mine][:, None]
            weights = scores.masked_fill(~visible[None], float("-inf")).softmax(dim=-1)
            out[mine] = torch.einsum("htl,lhd->thd", weights, V)
        return self.out(out.reshape(t, -1))


# %% [block] Transformer block
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 896 → 4864 → 896."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, batch, pool, layer):
        x = x + self.attention(self.norm1(x), batch, pool, layer)
        return x + self.mlp(self.norm2(x))


# %% [output] Next token
class Output(nn.Module):
    """Final RMSNorm, then the vocabulary projection (tied to the embedding), for each request's last token only."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(x))


# %% [model] One forward over the whole step
@dataclass
class Scheduled:
    """One request's share of a step: \`count\` tokens at positions length − count … length − 1, sitting at \`start\` in
    the step's flat list of tokens; \`table\` is its block table during the step."""

    request: "Request"
    start: int
    count: int
    table: list
    length: int


class Batch:
    """The step's tokens, flattened: ids and positions per token, the slot each token's K and V go to, where each
    request's tokens sit in the flat list, and the block tables attention reads (built once per step, as a server
    prepares its attention metadata)."""

    def __init__(self, work, pool, device):
        ids, positions, slots, self.seqs, offset = [], [], [], [], 0
        for r, start, count in work:
            ids += r.tokens()[start : start + count]
            positions += range(start, start + count)
            slots.append(pool.slots(r.table, start, start + count))
            self.seqs.append(Scheduled(r, offset, count, list(r.table), start + count))
            offset += count
        self.ids = torch.tensor(ids, device=device)
        self.positions = torch.tensor(positions, device=device)
        self.slots = torch.cat(slots).to(device)
        self.last = torch.tensor([s.start + s.count - 1 for s in self.seqs], device=device)
        decodes = [s for s in self.seqs if s.count == 1]
        self.decode_rows = torch.tensor([s.start for s in decodes], device=device) if decodes else None
        if decodes:
            width = max(len(s.table) for s in decodes)  # pad each table by repeating its last block
            self.decode_tables = torch.tensor([s.table + s.table[-1:] * (width - len(s.table)) for s in decodes], device=device)
            self.decode_lengths = torch.tensor([s.length for s in decodes], device=device)
        self.prefills = [(slice(s.start, s.start + s.count), torch.tensor(s.table, device=device), s.length)
                         for s in self.seqs if s.count > 1]


class Qwen2(nn.Module):
    """The step's tokens go through every layer together; logits come out for each request's last token."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, batch, pool):
        x = self.embedding(batch.ids)
        for layer, block in enumerate(self.blocks):
            x = block(x, batch, pool, layer)
        return self.output(x[batch.last])  # (requests, vocab)


# %% [scheduler] The scheduler
@dataclass
class Request:
    id: str
    prompt: list
    max_tokens: int
    output: list = field(default_factory=list)
    table: list = field(default_factory=list)
    computed: int = 0  # positions whose K and V are in the cache
    preempted: int = 0

    def tokens(self):
        return self.prompt + self.output


class Scheduler:
    """vLLM's step loop. Every running request first gets its next token (a new block when its last one is full; if
    the pool is empty, the most recently admitted request is preempted: its blocks are freed and it waits to be
    recomputed). Then waiting requests join, oldest first, with their whole prompt, while a seat (max_num_seqs), the
    token budget and the blocks for the prompt allow. Static batching admits only into an empty batch."""

    def __init__(self, c: Config, pool: BlockPool, policy="continuous"):
        self.c, self.pool, self.policy = c, pool, policy
        self.waiting, self.running, self.finished = deque(), [], []

    def add(self, request):
        self.waiting.append(request)

    def schedule(self):
        work, budget = [], self.c.max_num_batched_tokens
        for r in list(self.running):
            if r not in self.running:  # preempted below
                continue
            while not self.pool.allocate(r.table, r.computed + 1):
                victim = self.running.pop()
                self.pool.release(victim.table)
                victim.computed, victim.preempted = 0, victim.preempted + 1
                self.waiting.appendleft(victim)
                if victim is r:
                    break
            if r in self.running:
                work.append((r, r.computed, 1))
                budget -= 1
        if self.policy == "static" and self.running:
            return work
        while self.waiting and len(self.running) < self.c.max_num_seqs:
            r = self.waiting[0]
            count = len(r.tokens()) - r.computed  # a recomputed request brings its output back too
            if count > budget or not self.pool.allocate(r.table, len(r.tokens())):
                break
            self.waiting.popleft()
            self.running.append(r)
            work.append((r, r.computed, count))
            budget -= count
        return work


# %% [step] One engine step
@torch.no_grad()
def step(model, scheduler, pool, run=None):
    """Schedule, run one forward over every scheduled token, and give each request whose whole sequence is now cached
    its next greedy token. Finished requests (end of turn or max_tokens) leave and free their blocks at once.
    \`run(fn)\` may wrap the forward, e.g. to time it. Returns the step's Batch (None when nothing was scheduled)."""
    work = scheduler.schedule()
    if not work:
        return None
    batch = Batch(work, pool, pool.k.device)
    logits = (run or (lambda fn: fn()))(lambda: model(batch, pool))
    for (r, start, count), row in zip(work, logits):
        r.computed = start + count
        if r.computed == len(r.tokens()):
            r.output.append(int(row.argmax()))
            if r.output[-1] == scheduler.c.eos or len(r.output) == r.max_tokens:
                scheduler.running.remove(r)
                pool.release(r.table)
                scheduler.finished.append(r)
    return batch


# %% [serve] Serve a set of requests
@torch.no_grad()
def serve(model, requests, c: Config, policy="continuous", device="cpu", run=None, log=None):
    """Every request arrives at step 0 and waits; steps run until all have finished. Returns the requests; \`log(batch,
    scheduler, pool)\` sees every step."""
    pool = BlockPool(c, device)
    scheduler = Scheduler(c, pool, policy)
    for r in requests:
        scheduler.add(r)
    while scheduler.waiting or scheduler.running:
        batch = step(model, scheduler, pool, run)
        if batch is None:
            raise RuntimeError(f"{scheduler.waiting[0].id} can never fit in the KV pool")
        if log is not None:
            log(batch, scheduler, pool)
    return requests


def alone(model, request, c: Config, device="cpu"):
    """The same request served by itself: the reference its batched output must equal."""
    solo = Request(request.id, list(request.prompt), request.max_tokens)
    return serve(model, [solo], c, device=device)[0].output
`;export{e as default};
