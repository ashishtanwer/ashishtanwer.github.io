const e=`# The KV cache in Qwen2.5-0.5B-Instruct, the model in the KV-cache tab, written as plain PyTorch blocks.
# The prefill writes every prompt token's keys and values in every layer; each decode step feeds one
# token, writes one row per layer and attends over all of them. Each "# %% [id]" section is one block.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Qwen2.5-0.5B-Instruct
    vocab_size: int = 151936
    d_model: int = 896
    n_layers: int = 24
    n_heads: int = 14  # 64 dimensions per head
    n_kv_heads: int = 2  # grouped-query attention: seven query heads share each key/value head
    d_mlp: int = 4864
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    bytes_per_element: int = 2  # the checkpoint's bfloat16
    block_size: int = 16  # paged cache: positions per block (vLLM's default)

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


# %% [rope] RoPE at the tokens' positions
def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d). A decode step's
    one new key is rotated by its true position, so cached keys never need rotating again.

    x: (heads, seq, d). positions: (seq,) absolute positions."""
    d = x.shape[-1]
    frequencies = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device).float() / d)
    angles = positions[:, None].float() * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [kv-cache] KV cache
class KVCache:
    """Per layer, every position's key (after RoPE) and value so far: (n_kv_heads, seq, head_dim).
    The prefill appends all prompt rows at once; each decode step appends exactly one."""

    def __init__(self, c: Config):
        self.k = [None] * c.n_layers
        self.v = [None] * c.n_layers

    def append(self, layer, k, v):  # k, v: (n_kv_heads, new, head_dim)
        if self.k[layer] is not None:
            k, v = torch.cat((self.k[layer], k), dim=1), torch.cat((self.v[layer], v), dim=1)
        self.k[layer], self.v[layer] = k, v
        return k, v  # every row so far, for this step's queries to read

    def __len__(self):
        return 0 if self.k[0] is None else self.k[0].shape[1]


# %% [attention] Attention over the cache
class Attention(nn.Module):
    """Grouped-query attention reading the cache. Only the new tokens are projected; their queries
    attend to every cached position and to the new positions up to their own."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def forward(self, x, positions, cache, layer):  # x: (new, d_model); positions: (new,)
        c, t = self.c, x.shape[0]
        q = self.q(x).view(t, c.n_heads, c.head_dim).transpose(0, 1)  # (heads, new, head_dim)
        k = self.k(x).view(t, c.n_kv_heads, c.head_dim).transpose(0, 1)
        v = self.v(x).view(t, c.n_kv_heads, c.head_dim).transpose(0, 1)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        k, v = cache.append(layer, k, v)  # (kv_heads, seq, head_dim), seq = cached + new
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=0), v.repeat_interleave(group, dim=0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)  # (heads, new, seq)
        visible = torch.arange(k.shape[1], device=x.device) <= positions[:, None]  # causal
        weights = scores.masked_fill(~visible, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(0, 1).reshape(t, -1)), weights


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

    def forward(self, x, positions, cache, layer):
        x = x + self.attention(self.norm1(x), positions, cache, layer)[0]
        return x + self.mlp(self.norm2(x))


# %% [output] Next token
class Output(nn.Module):
    """Final RMSNorm, then the vocabulary projection, which shares the token embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(x))  # (new, vocab)


# %% [model] Qwen2.5 with a KV cache
class Qwen2(nn.Module):
    """Runs only the new tokens: all prompt tokens at the prefill, one token per decode step."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.weight  # weight tying

    def forward(self, ids, positions, cache):  # ids, positions: (new,)
        x = self.embedding(ids)
        for layer, block in enumerate(self.blocks):
            x = block(x, positions, cache, layer)
        return self.output(x)  # logits: (new, vocab)


# %% [generate] Prefill, then one token per step
@torch.no_grad()
def generate(model, prompt, max_new_tokens=24, eos=151645, cache=None):
    """Greedy decoding. The prefill writes all prompt rows; each later step feeds the last token,
    writes one row per layer and reads the whole cache. Returns the tokens and the cache."""
    cache = KVCache(model.c) if cache is None else cache
    logits = model(prompt, torch.arange(len(prompt)), cache)[-1]  # prefill
    tokens = []
    while len(tokens) < max_new_tokens:
        token = logits.argmax()
        tokens.append(int(token))
        if token == eos or len(tokens) == max_new_tokens:
            break
        logits = model(token[None], torch.tensor([len(cache)]), cache)[-1]  # one new row per layer
    return tokens, cache


@torch.no_grad()
def generate_without_cache(model, prompt, max_new_tokens=24, eos=151645):
    """The same answer the slow way: every step runs the whole sequence again (a fresh cache each time)."""
    ids, tokens = prompt, []
    while len(tokens) < max_new_tokens:
        token = model(ids, torch.arange(len(ids)), KVCache(model.c))[-1].argmax()
        tokens.append(int(token))
        if token == eos or len(tokens) == max_new_tokens:
            break
        ids = torch.cat((ids, token[None]))
    return tokens


# %% [kv-bytes] The cache's size
def cache_bytes(c: Config, positions: int, layers=None):
    """K and V for every key/value head: 2 × 2 heads × 64 × 2 bytes = 512 bytes per position per layer,
    12 KiB per position over 24 layers. With 14 key/value heads (plain multi-head) it would be 7×."""
    layers = c.n_layers if layers is None else layers
    return 2 * layers * c.n_kv_heads * c.head_dim * positions * c.bytes_per_element


# %% [flops] FLOPs with and without the cache
def step_flops(c: Config, length: int, new: int):
    """Matmul FLOPs (2 per multiply-add) of a forward over \`new\` tokens ending at cache length \`length\`,
    with logits for the last position only. With the cache a decode step has new = 1; without it,
    new = length: every token goes through every layer again. Attention scores every new query against
    all \`length\` keys and masks the later ones afterwards, as eager attention does."""
    per_token = c.d_model * c.n_heads * c.head_dim + 2 * c.d_model * c.n_kv_heads * c.head_dim  # q, k, v
    per_token += c.n_heads * c.head_dim * c.d_model + 3 * c.d_model * c.d_mlp  # o, gate, up, down
    attention = 2 * c.n_heads * c.head_dim * new * length  # q · k, then weights · v
    return 2 * c.n_layers * (per_token * new + attention) + 2 * c.d_model * c.vocab_size


# %% [paged] Paged cache
class PagedKVCache:
    """vLLM's layout: per layer, a pool of fixed-size blocks (blocks, block_size, kv_heads, head_dim)
    for K and for V, and per sequence a block table: logical block i lives in physical block table[i].
    A block is taken from the free list only when the last one fills, so at most block_size − 1 slots
    are allocated and empty, however long the answer may grow."""

    def __init__(self, c: Config, num_blocks: int):
        shape = (c.n_layers, num_blocks, c.block_size, c.n_kv_heads, c.head_dim)
        self.c, self.k, self.v = c, torch.zeros(shape), torch.zeros(shape)
        self.free = list(range(num_blocks))
        self.table = []
        self.lengths = [0] * c.n_layers

    def append(self, layer, k, v):  # k, v: (n_kv_heads, new, head_dim)
        size, start = self.c.block_size, self.lengths[layer]
        end = start + k.shape[1]
        while len(self.table) * size < end:
            self.table.append(self.free.pop(0))
        slots = torch.tensor([self.table[p // size] * size + p % size for p in range(start, end)])
        self.k[layer].view(-1, self.c.n_kv_heads, self.c.head_dim)[slots] = k.transpose(0, 1)
        self.v[layer].view(-1, self.c.n_kv_heads, self.c.head_dim)[slots] = v.transpose(0, 1)
        self.lengths[layer] = end
        return self.gather(layer)

    def gather(self, layer):
        """The sequence's rows in order, read through the block table: (n_kv_heads, seq, head_dim)."""
        blocks = torch.tensor(self.table)
        k = self.k[layer][blocks].flatten(0, 1)[: self.lengths[layer]]
        v = self.v[layer][blocks].flatten(0, 1)[: self.lengths[layer]]
        return k.transpose(0, 1), v.transpose(0, 1)

    def __len__(self):
        return self.lengths[0]


# tokens, cache = generate(Qwen2(), prompt_ids)  # prompt_ids from Qwen's chat template
# tokens, paged = generate(Qwen2(), prompt_ids, cache=PagedKVCache(Config(), num_blocks=8))
`;export{e as default};
