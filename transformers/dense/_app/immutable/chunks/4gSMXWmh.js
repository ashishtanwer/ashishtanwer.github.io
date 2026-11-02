const e=`# Tensor parallelism for GPT-2 (124M), the model in the TP tab, as plain PyTorch blocks.
# Megatron-LM's split: attention's QKV matrix and the MLP's first matrix are cut by output columns, the
# attention output and the MLP's second matrix by input rows, so each half of a block needs one all-reduce
# forward and one backward. Every rank is its own process running this code; \`comm\` is torch.distributed
# (the recorder passes one that also logs each call). Each "# %% [id]" section is one block.
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
    ranks: int = 4  # tensor-parallel size: 3 heads and 768 MLP columns per rank

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


class Comm:
    """What a rank calls to talk to the others."""

    def all_reduce(self, x, label=""):
        dist.all_reduce(x)  # in place: every rank ends with the sum over all ranks
        return x


# %% [f-g] f and g: the two all-reduces
class Copy(torch.autograd.Function):
    """Megatron's f, in front of a column-parallel layer. Forward it is the identity: every rank already
    has x. Backward each rank's gradient for x comes only from its own columns, so they are summed."""

    @staticmethod
    def forward(ctx, x, comm, label):
        ctx.comm, ctx.label = comm, label
        return x

    @staticmethod
    def backward(ctx, grad):
        return ctx.comm.all_reduce(grad.contiguous().clone(), ctx.label), None, None


class Reduce(torch.autograd.Function):
    """Megatron's g, after a row-parallel layer. Forward the ranks' partial sums are added; backward it is
    the identity, since every rank needs the whole gradient of the sum."""

    @staticmethod
    def forward(ctx, x, comm, label):
        return comm.all_reduce(x.contiguous().clone(), label)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


# %% [column-parallel] Column-parallel linear
class ColumnParallel(nn.Module):
    """This rank's output columns of x @ W + b. GPT-2 stores W as (in, out); the rank keeps W[:, cols],
    (in, out / ranks), and b[cols]. Its output stays split; nothing is sent forward."""

    def __init__(self, d_in, d_out, comm, label):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.empty(d_out))
        self.comm, self.label = comm, label

    def forward(self, x):  # x: (seq, d_in), the same on every rank
        return Copy.apply(x, self.comm, self.label) @ self.weight + self.bias


# %% [row-parallel] Row-parallel linear
class RowParallel(nn.Module):
    """x @ W split by W's input rows: the rank holds W[rows, :], (in / ranks, out), and the matching slice
    of x, so x_r @ W_r is its share of the product. The all-reduce adds the shares; the bias, kept on every
    rank, is added once after the sum."""

    def __init__(self, d_in, d_out, comm, label):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.empty(d_out))
        self.comm, self.label = comm, label

    def forward(self, x):  # x: (seq, d_in / ranks), this rank's slice
        return Reduce.apply(x @ self.weight, self.comm, self.label) + self.bias


# %% [attention] Attention with this rank's heads
class Attention(nn.Module):
    """Heads are independent, so a rank computes 12 / ranks of them from start to end: its Q, K and V
    columns (column-parallel), causal softmax attention, and its rows of the output projection
    (row-parallel)."""

    def __init__(self, c: Config, comm, layer):
        super().__init__()
        self.h, self.d = c.n_heads // c.ranks, c.head_dim
        self.c_attn = ColumnParallel(c.d_model, 3 * self.h * self.d, comm, f"{layer}/attention/f")
        self.c_proj = RowParallel(self.h * self.d, c.d_model, comm, f"{layer}/attention/g")

    def forward(self, x):  # x: (seq, 768) → (seq, 768), already summed over ranks
        t = x.shape[0]
        q, k, v = self.c_attn(x).view(t, 3, self.h, self.d).permute(1, 2, 0, 3)  # each (heads, seq, 64)
        scores = q @ k.transpose(-1, -2) / self.d**0.5
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        self.weights = scores.masked_fill(~causal, float("-inf")).softmax(-1)  # (heads, seq, seq)
        y = (self.weights @ v).transpose(0, 1).reshape(t, self.h * self.d)
        return self.c_proj(y)


# %% [mlp] MLP with this rank's columns
class MLP(nn.Module):
    """The first matrix's 3072 columns are split (column-parallel), so GELU runs on the rank's own
    columns with no communication; the second matrix's rows follow them (row-parallel)."""

    def __init__(self, c: Config, comm, layer):
        super().__init__()
        self.c_fc = ColumnParallel(c.d_model, c.d_mlp // c.ranks, comm, f"{layer}/mlp/f")
        self.c_proj = RowParallel(c.d_mlp // c.ranks, c.d_model, comm, f"{layer}/mlp/g")

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))  # GPT-2's gelu_new


# %% [block] One block, LayerNorms on every rank
class Block(nn.Module):
    """The LayerNorms and residual adds are cheap and run on every rank on the whole (seq, 768) tensor;
    after each all-reduce every rank has the same values."""

    def __init__(self, c: Config, comm, layer):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attn = Attention(c, comm, layer)
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c, comm, layer)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


# %% [shards] What each rank loads
def shard(c: Config, rank: int, name: str):
    """The slice of checkpoint tensor \`name\` that \`rank\` keeps, as (dim, [(start, stop), ...]), or None
    for a tensor every rank keeps whole. The names are the checkpoint's own. c_attn's 2304 columns are
    Q | K | V, 768 each, so a rank takes its three heads' 192 columns from each of the three."""
    h, m = c.d_model // c.ranks, c.d_mlp // c.ranks  # 192 attention columns, 768 MLP columns
    layer = name.split(".", 2)[-1] if name.startswith("h.") else name
    columns = 1 if layer.endswith("weight") else 0  # W is (in, out): columns are dim 1; a bias has one dim
    if layer.startswith("attn.c_attn."):
        return columns, [(p * c.d_model + rank * h, p * c.d_model + (rank + 1) * h) for p in range(3)]
    if layer.startswith("mlp.c_fc."):
        return columns, [(rank * m, (rank + 1) * m)]
    if layer == "attn.c_proj.weight":  # rows; its bias is kept whole
        return 0, [(rank * h, (rank + 1) * h)]
    if layer == "mlp.c_proj.weight":
        return 0, [(rank * m, (rank + 1) * m)]
    return None


def take(full, part):
    """Applies a shard to a whole tensor."""
    if part is None:
        return full
    dim, ranges = part
    return torch.cat([full.narrow(dim, a, b - a) for a, b in ranges], dim)


# %% [model] GPT-2 on one rank
class GPT2Rank(nn.Module):
    """The whole model as one rank runs it. The embeddings, LayerNorms and the tied LM head are kept
    whole on every rank here (Megatron also splits the vocabulary); every rank ends with the same logits."""

    def __init__(self, c: Config, comm):
        super().__init__()
        self.wte = nn.Embedding(c.vocab, c.d_model)
        self.wpe = nn.Embedding(c.context, c.d_model)
        self.h = nn.ModuleList(Block(c, comm, i) for i in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, ids):  # ids: (seq,)
        x = self.wte(ids) + self.wpe(torch.arange(len(ids)))
        for block in self.h:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T  # (seq, vocab)

    def load(self, c: Config, rank: int, read):
        """read(name, part) returns that slice of the checkpoint tensor, read from disk."""
        state = {}
        for name in self.state_dict():
            state[name] = read(name, shard(c, rank, name))
        self.load_state_dict(state, strict=True)
        return self
`;export{e as default};
