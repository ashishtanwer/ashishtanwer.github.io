const n=`# Context parallelism for GPT-2 (124M), the model in the CP tab, as plain PyTorch blocks.
# A long sequence is split across ranks; every rank keeps the whole model and runs it on its own tokens.
# Everything but attention is per token. For attention, ring attention passes each rank's keys and values
# around the ring of ranks: at each step a rank attends from its queries to the block it holds, keeps
# softmax statistics (the row maximum m and the sum l), and merges them online, so no rank ever holds the
# whole score matrix. Every rank is its own process; \`comm\` posts sends and receives that travel while the
# rank computes. Each "# %% [id]" section is one block.
import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # GPT-2 (124M)
    d_model: int = 768
    n_heads: int = 12
    d_mlp: int = 3072
    n_layers: int = 12
    vocab: int = 50257
    context: int = 1024
    eps: float = 1e-5
    ranks: int = 4
    split: str = "contiguous"  # or "zigzag"

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class Comm:
    """What a rank calls: post() starts sends and receives and returns at once, finish() waits for them."""

    def __init__(self):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()

    def post(self, sends, recvs):  # sends: [(tensor, dst, label)], recvs: [(shape, src, label)]
        out = [torch.empty(shape) for shape, _, _ in recvs]
        ops = [dist.P2POp(dist.isend, t.contiguous(), dst) for t, dst, _ in sends]
        ops += [dist.P2POp(dist.irecv, buf, src) for buf, (_, src, _) in zip(out, recvs)]
        return dist.batch_isend_irecv(ops), out

    def finish(self, handle):
        works, out = handle
        for work in works:
            work.wait()
        return out

    def span(self, op, label=""):  # the recorder's Comm times this
        return nullcontext()


# %% [shards] Which tokens each rank holds
def chunks(c: Config, rank: int):
    """Rank \`rank\`'s tokens as [(slice of its local tensor, positions in the whole sequence)]. Contiguous:
    one chunk, the rank's quarter. Zigzag (the load-balanced split of Megatron's context parallelism and
    Llama 3): the sequence in 2 × ranks chunks, rank r taking chunk r and chunk 2·ranks − 1 − r, so under
    the causal mask every rank has the same work."""
    n = c.context // c.ranks
    if c.split == "contiguous":
        return [(slice(0, n), torch.arange(rank * n, (rank + 1) * n))]
    h = n // 2
    a, b = rank, 2 * c.ranks - 1 - rank
    return [(slice(0, h), torch.arange(a * h, (a + 1) * h)), (slice(h, n), torch.arange(b * h, (b + 1) * h))]


def positions(c: Config, rank: int) -> torch.Tensor:
    return torch.cat([pos for _, pos in chunks(c, rank)])


# %% [block-attention] One query block against one key block
def block_attention(q, k, v, q_pos, k_pos):
    """q: (heads, n, 64); k, v: (heads, n_k, 64); positions in the whole sequence. Returns the block's
    unnormalised output o = Σ_j e^{s_ij − m_i} v_j, its row maximum m and row sum l = Σ_j e^{s_ij − m_i},
    with the causal mask applied by position. A row with no visible key has m = −∞ and l = 0."""
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])  # (heads, n, n_k)
    s = s.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))
    m = s.amax(-1)
    p = torch.exp(s - torch.where(torch.isinf(m), 0.0, m)[..., None])
    return p @ v, m, p.sum(-1)


# %% [merge] The online-softmax merge
def merge(o, m, l, o_b, m_b, l_b):
    """Adds a block to the running (o, m, l): both are rescaled to the new maximum, so
    l = l·e^{m − m'} + l_b·e^{m_b − m'} and o likewise. The order of blocks does not matter."""
    m_new = torch.maximum(m, m_b)
    safe = torch.where(torch.isinf(m_new), 0.0, m_new)
    a, b = torch.exp(m - safe), torch.exp(m_b - safe)
    return o * a[..., None] + o_b * b[..., None], m_new, l * a + l_b * b


# %% [ring] Ring attention
def ring_attention(c: Config, q, k, v, comm, label="", record=None):
    """Step t: this rank holds the keys and values that started on rank (rank − t) mod ranks. It sends
    them on to rank + 1 and receives the next block from rank − 1 while it computes; after \`ranks\` steps it
    has seen every block. Only (query chunk, key chunk) pairs with a visible key are computed. \`record\`,
    if given, gets each step's (source, o_b, m_b, l_b)."""
    rank, world = comm.rank, comm.world
    o, m, l = torch.zeros_like(q), torch.full(q.shape[:-1], float("-inf")), torch.zeros(q.shape[:-1])
    kv = torch.stack([k, v])  # (2, heads, n, 64): what travels
    for step in range(world):
        src = (rank - step) % world
        if step < world - 1:  # send on what we hold, receive the next block, while computing
            handle = comm.post([(kv, (rank + 1) % world, f"{label}/{step}")], [(kv.shape, (rank - 1) % world, f"{label}/{step}")])
        with comm.span("block", f"{label}/{step}"):
            o_b, m_b, l_b = torch.zeros_like(o), torch.full(m.shape, float("-inf")), torch.zeros_like(l)
            for qs, q_pos in chunks(c, rank):
                for ks, k_pos in chunks(c, src):
                    if k_pos[0] <= q_pos[-1]:  # some key is not in the future (positions ascend)
                        part = block_attention(q[:, qs], kv[0][:, ks], kv[1][:, ks], q_pos, k_pos)
                        o_b[:, qs], m_b[:, qs], l_b[:, qs] = merge(o_b[:, qs], m_b[:, qs], l_b[:, qs], *part)
            o, m, l = merge(o, m, l, o_b, m_b, l_b)
        if record is not None:
            record.append((src, o_b, m_b, l_b))
        if step < world - 1:
            kv = comm.finish(handle)[0]
    return o / l[..., None]  # (heads, n, 64)


# %% [block] One block on one rank's tokens
class Block(nn.Module):
    """GPT-2's block. The LayerNorms, the projections and the MLP act on each token alone, so they run
    on the rank's own tokens unchanged; only attention needs the other ranks."""

    def __init__(self, c: Config, comm, layer):
        super().__init__()
        self.c, self.comm, self.layer = c, comm, layer
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attn = nn.ModuleDict(dict(c_attn=Conv1D(c.d_model, 3 * c.d_model), c_proj=Conv1D(c.d_model, c.d_model)))
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = nn.ModuleDict(dict(c_fc=Conv1D(c.d_model, c.d_mlp), c_proj=Conv1D(c.d_mlp, c.d_model)))
        self.record = None

    def forward(self, x):  # x: (n, 768), this rank's tokens
        n, c = x.shape[0], self.c
        with self.comm.span("local", f"{self.layer}"):
            q, k, v = self.attn.c_attn(self.ln_1(x)).view(n, 3, c.n_heads, c.head_dim).permute(1, 2, 0, 3)
        y = ring_attention(c, q, k, v, self.comm, f"{self.layer}", self.record)
        with self.comm.span("mlp", f"{self.layer}"):
            self.attended = y.transpose(0, 1).reshape(n, -1)
            x = x + self.attn.c_proj(self.attended)
            return x + self.mlp.c_proj(F.gelu(self.mlp.c_fc(self.ln_2(x)), approximate="tanh"))


class Conv1D(nn.Module):
    """GPT-2's linear layer: x @ W + b with W stored as (in, out)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.empty(d_out))

    def forward(self, x):
        return x @ self.weight + self.bias


# %% [model] GPT-2 on one rank's shard of the sequence
class GPT2Shard(nn.Module):
    """The whole model on every rank, run on the rank's tokens with their positions in the full sequence."""

    def __init__(self, c: Config, comm):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab, c.d_model)
        self.wpe = nn.Embedding(c.context, c.d_model)
        self.h = nn.ModuleList(Block(c, comm, i) for i in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, ids, rank):  # ids: the whole sequence's tokens; the rank takes its own
        pos = positions(self.c, rank)
        x = self.wte(ids[pos]) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        return self.ln_f(x)  # (n, 768); the logits are this @ wte.weight.T
`;export{n as default};
