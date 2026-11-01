const e=`# Chunked prefill, the method in the Efficiency Explainer's Chunked prefill tab, written as plain PyTorch blocks.
# A step may run at most a token budget of tokens. Running requests come first, each decode costing one token; what is
# left of the budget goes to prompts, and a prompt longer than that is split into chunks prefilled over several steps
# (vLLM, Sarathi-Serve). Without chunking a long prompt runs whole in one step, and every request decoding beside it
# waits that step out. A chunk's queries attend to the keys and values earlier chunks left in the cache, so the result
# is the same as one prefill. Shown on Qwen2.5-0.5B-Instruct. Each "# %% [id]" section is one block of the page.
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
    num_blocks: int = 192  # the KV pool every request allocates from
    max_num_seqs: int = 8  # requests running at once
    max_num_batched_tokens: int = 256  # the token budget: tokens one step may run
    chunked_prefill: bool = True  # split a prompt that does not fit the budget left in a step
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
    request's slots; then each request's queries read that request's blocks only. A chunk of a prompt is a prefill
    that starts at a later position: its queries read the blocks earlier chunks wrote, then their own rows, causally.
    Decodes (one query each) are gathered into one padded batch, as a paged-attention kernel reads them."""

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
        for mine, blocks, length in batch.prefills:  # a prompt or a chunk of one: earlier chunks' rows, then causal
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


# %% [budget] A token budget per step
@dataclass
class Request:
    id: str
    prompt: list
    max_tokens: int
    arrival: int = 0  # the step at which the request reaches the server
    output: list = field(default_factory=list)
    table: list = field(default_factory=list)
    computed: int = 0  # positions whose K and V are in the cache
    preempted: int = 0

    def tokens(self):
        return self.prompt + self.output


class Scheduler:
    """vLLM's scheduler with a token budget. Running requests come first: a decode takes one token, a prompt part-way
    through its prefill takes its next chunk, as much as the budget has left. Then waiting requests join, oldest first,
    while seats, budget and blocks allow; with chunked_prefill a prompt starts with whatever the budget has left,
    without it the whole prompt must fit. If a running request needs a block and the pool is empty, the most recently
    admitted request is preempted: its blocks are freed and it waits to be recomputed."""

    def __init__(self, c: Config, pool: BlockPool):
        self.c, self.pool = c, pool
        self.waiting, self.running, self.finished = deque(), [], []

    def add(self, request):
        self.waiting.append(request)

    def schedule(self):
        work, budget = [], self.c.max_num_batched_tokens
        for r in list(self.running):
            if r not in self.running or budget == 0:  # preempted below, or no budget left this step
                continue
            count = min(len(r.tokens()) - r.computed, budget)
            while not self.pool.allocate(r.table, r.computed + count):
                victim = self.running.pop()
                self.pool.release(victim.table)
                victim.computed, victim.preempted = 0, victim.preempted + 1
                self.waiting.appendleft(victim)
                if victim is r:
                    break
            if r in self.running:
                work.append((r, r.computed, count))
                budget -= count
        while self.waiting and len(self.running) < self.c.max_num_seqs and budget > 0:
            r = self.waiting[0]
            count = len(r.tokens()) - r.computed
            if self.c.chunked_prefill:
                count = min(count, budget)
            if count > budget or not self.pool.allocate(r.table, r.computed + count):
                break
            self.waiting.popleft()
            self.running.append(r)
            work.append((r, r.computed, count))
            budget -= count
        return work


# %% [step] One engine step
@torch.no_grad()
def step(model, scheduler, pool, run=None):
    """Schedule, run one forward over every scheduled token (decodes and chunks together), and give each request whose
    whole sequence is now cached its next greedy token; a chunk that leaves part of its prompt for later samples
    nothing. Finished requests leave and free their blocks. \`run(fn)\` may wrap the forward, e.g. to time it. Returns
    the step's Batch (None when nothing was scheduled)."""
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


# %% [serve] Serve requests as they arrive
@torch.no_grad()
def serve(model, requests, c: Config, device="cpu", run=None, log=None):
    """Each request joins the waiting queue at the step it arrives; steps run until all have finished. Returns the
    requests; \`log(batch, scheduler, pool)\` sees every step (batch is None when nothing could run)."""
    pool = BlockPool(c, device)
    scheduler = Scheduler(c, pool)
    pending = sorted(requests, key=lambda r: r.arrival)
    clock = 0
    while pending or scheduler.waiting or scheduler.running:
        while pending and pending[0].arrival <= clock:
            scheduler.add(pending.pop(0))
        batch = step(model, scheduler, pool, run)
        if batch is None and not scheduler.running and (not pending or pending[0].arrival > clock + 1) and scheduler.waiting:
            raise RuntimeError(f"{scheduler.waiting[0].id} can never fit in the KV pool")
        if log is not None:
            log(batch, scheduler, pool)
        clock += 1
    return requests


def alone(model, request, c: Config, device="cpu"):
    """The same request served by itself: the reference its output must equal, with or without chunks."""
    solo = Request(request.id, list(request.prompt), request.max_tokens)
    return serve(model, [solo], c, device=device)[0].output


# %% [prefill] One prompt, whole or in chunks
@torch.no_grad()
def prefill(model, prompt, c: Config, chunk=None, device="cpu"):
    """Prefill one prompt in a single forward, or chunk tokens at a time, each chunk reading what the earlier ones
    cached. Returns the last position's logits, and every layer's K and V in position order (gathered through the
    block table)."""
    pool, r = BlockPool(c, device), Request("prompt", list(prompt), 1)
    chunk = chunk or len(prompt)
    for start in range(0, len(prompt), chunk):
        count = min(chunk, len(prompt) - start)
        pool.allocate(r.table, start + count)
        logits = model(Batch([(r, start, count)], pool, device), pool)[0]
    blocks = torch.tensor(r.table, device=pool.k.device)
    k = pool.k[:, blocks].flatten(1, 2)[:, : len(prompt)]  # (layers, positions, kv_heads, head_dim)
    v = pool.v[:, blocks].flatten(1, 2)[:, : len(prompt)]
    return logits, k, v
`;export{e as default};
