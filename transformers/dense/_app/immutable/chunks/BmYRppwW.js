const e=`# Block-sparse attention as in BigBird-RoBERTa base, the model in the Attention Explainer's Sparse tab.
# Each "# %% [id]" section is one step of the diagram. Tokens come in blocks of 64; each block of
# queries reads the 2 global blocks, its 3-block window and 3 random blocks instead of all of them.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 768
    n_heads: int = 12  # 768 / 12 = 64 dimensions per head
    block: int = 64  # tokens per block: 1,024 tokens are 16 blocks
    n_random: int = 3  # random key blocks per query block
    eps: float = 1e-12


# %% [qkv] Q · K · V projection
class QKV(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.query = nn.Linear(c.d_model, c.d_model)
        self.key = nn.Linear(c.d_model, c.d_model)
        self.value = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model); seq is a whole number of blocks
        b, t, d = x.shape
        split = lambda z: z.view(b, t, self.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, d_head)
        return split(self.query(x)), split(self.key(x)), split(self.value(x))


# %% [pattern] Which key blocks each query block reads
def block_pattern(n_blocks, random_blocks):
    """How many times each query block gathers each key block, per head: (heads, n_blocks, n_blocks).

    The first and last blocks are global: they read every block, and every block reads them. Every
    other block also reads its window (itself and both neighbours) and its random blocks.
    random_blocks (heads, n_blocks - 2, n_random) lists the random key blocks of query blocks
    1 … n_blocks − 2; BigBird draws them once per layer (numpy, seed = layer index). A random block
    that is already global or in the window is gathered again, so it counts twice."""
    heads = random_blocks.shape[0]
    last = n_blocks - 1
    count = torch.zeros(heads, n_blocks, n_blocks)
    count[:, 0] = count[:, last] = 1  # global queries
    for i in range(1, last):
        count[:, i, sorted({0, i - 1, i, i + 1, last})] = 1  # global keys and the window
        for head in range(heads):
            for j in random_blocks[head, i - 1].tolist():
                count[head, i, j] += 1
    return count


# %% [attention] Block-sparse attention
def block_sparse_attention(q, k, v, count, block):
    """Each query takes a softmax over the keys of the blocks its block reads. Written densely
    for clarity: the real kernel gathers only those blocks, so its cost grows like
    seq · (5 + n_random) · block instead of seq². Adding log(count) to a score masks a block that
    is not read (log 0 = −inf) and doubles a key whose block is gathered twice."""
    copies = count.repeat_interleave(block, -2).repeat_interleave(block, -1)  # (heads, seq, seq)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) + copies.log()
    weights = scores.softmax(dim=-1)
    return weights @ v, weights  # each head's output, and the weights the head card sums per block


# %% [output] Concatenate heads, project, add and norm
class Output(nn.Module):
    """W_O, then BERT-style post-norm: LayerNorm(x + W_O · heads)."""

    def __init__(self, c: Config):
        super().__init__()
        self.dense = nn.Linear(c.d_model, c.d_model)
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, heads, x):  # heads: (batch, heads, seq, d_head); x: the layer's input
        b, h, t, d = heads.shape
        return self.norm(x + self.dense(heads.transpose(1, 2).reshape(b, t, h * d)))


# %% [model] BigBird attention layer
class SparseAttention(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.block = c.block
        self.qkv = QKV(c)
        self.output = Output(c)

    def forward(self, x, random_blocks):
        q, k, v = self.qkv(x)
        count = block_pattern(x.shape[1] // self.block, random_blocks)
        heads, _ = block_sparse_attention(q, k, v, count, self.block)
        return self.output(heads, x)
`;export{e as default};
