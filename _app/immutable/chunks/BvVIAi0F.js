const e=`# KV-cache eviction (H2O and SnapKV) on Qwen2.5-0.5B-Instruct, the model in the KV-eviction tab, as plain
# PyTorch. Each key/value head of each layer keeps a fixed budget of positions; evicted rows are gone, and
# the kept keys keep the RoPE rotation of their original positions. The model is the KV-cache tab's; only
# the cache changes. Each "# %% [id]" section is one block.
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
    n_kv_heads: int = 2  # seven query heads share each key/value head, and so share its kept positions
    d_mlp: int = 4864
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    budget: int = 32  # positions kept per key/value head per layer

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def group(self):
        return self.n_heads // self.n_kv_heads


# %% [evicting-cache] A cache that evicts
class EvictingCache:
    """Per layer: each key/value head's kept rows (n_kv_heads, kept, head_dim) and their original
    positions (n_kv_heads, kept). Heads keep different positions but the same number of them."""

    def __init__(self, c: Config):
        self.c = c
        self.k, self.v, self.positions = [None] * c.n_layers, [None] * c.n_layers, [None] * c.n_layers

    def append(self, layer, k, v, positions):  # k, v: (n_kv_heads, new, head_dim); positions: (new,)
        positions = positions.expand(k.shape[0], -1)
        if self.k[layer] is not None:
            k = torch.cat((self.k[layer], k), dim=1)
            v = torch.cat((self.v[layer], v), dim=1)
            positions = torch.cat((self.positions[layer], positions), dim=1)
        self.k[layer], self.v[layer], self.positions[layer] = k, v, positions
        return k, v, positions

    def keep(self, layer, index):  # index: (n_kv_heads, n) rows to keep, in order
        take = index[..., None].expand(-1, -1, self.c.head_dim)
        self.k[layer] = self.k[layer].gather(1, take)
        self.v[layer] = self.v[layer].gather(1, take)
        self.positions[layer] = self.positions[layer].gather(1, index)

    def observe(self, layer, weights):
        """weights: (n_heads, new, kept), this step's attention; a policy evicts here."""

    def __len__(self):  # positions written so far
        return 0 if self.positions[0] is None else int(self.positions[0].max()) + 1


# %% [h2o] H2O: heavy hitters + recent
def h2o_select(scores, heavy, recent):
    """scores: (n_kv_heads, kept), each row's attention accumulated over every query so far. Keeps the
    \`recent\` newest rows and the \`heavy\` older rows with the largest scores; returns their indices."""
    kept = scores.shape[1]
    older = scores.clone()
    older[:, kept - recent :] = float("-inf")
    newest = torch.arange(kept - recent, kept).expand(scores.shape[0], -1)
    return torch.cat((older.topk(heavy, dim=-1).indices, newest), dim=1).sort(dim=1).values


class H2OCache(EvictingCache):
    """The prefill attends to the whole prompt and then evicts down to the budget; every decode step
    adds one row and evicts one. A row's score is the attention it received, summed over every query
    so far and over the query heads that share its key/value head."""

    def __init__(self, c: Config, heavy=None, recent=None):
        super().__init__(c)
        self.heavy = c.budget // 2 if heavy is None else heavy
        self.recent = c.budget - self.heavy if recent is None else recent
        self.scores = [None] * c.n_layers

    def observe(self, layer, weights):
        received = weights.sum(dim=1).view(self.c.n_kv_heads, self.c.group, -1).sum(dim=1)
        old = self.scores[layer]
        if old is not None:
            received = received + F.pad(old, (0, received.shape[1] - old.shape[1]))
        self.scores[layer] = received
        if received.shape[1] > self.heavy + self.recent:
            index = h2o_select(received, self.heavy, self.recent)
            self.keep(layer, index)
            self.scores[layer] = received.gather(1, index)


# %% [snapkv] SnapKV: votes from an observation window
def snapkv_select(weights, group, budget, window=8, kernel=5):
    """weights: (n_heads, prompt, prompt), the prefill's attention. The last \`window\` queries vote for
    every earlier position with their attention; votes are summed over the query heads sharing a
    key/value head and smoothed by an average pool, so neighbours of a strong position come along.
    Keeps the top budget − window positions plus the window itself; returns their indices."""
    heads, _, length = weights.shape
    votes = weights[:, -window:, : length - window].sum(dim=1)
    votes = votes.view(heads // group, group, -1).sum(dim=1)  # (n_kv_heads, length − window)
    pooled = F.avg_pool1d(votes[:, None], kernel, stride=1, padding=kernel // 2)[:, 0]
    top = pooled.topk(budget - window, dim=-1).indices
    window_rows = torch.arange(length - window, length).expand(top.shape[0], -1)
    return torch.cat((top, window_rows), dim=1).sort(dim=1).values


class SnapKVCache(EvictingCache):
    """Compresses the prompt once, after the prefill; the answer's rows are then appended unevicted."""

    def __init__(self, c: Config, window=8, kernel=5):
        super().__init__(c)
        self.window, self.kernel, self.done = window, kernel, [False] * c.n_layers

    def observe(self, layer, weights):
        if not self.done[layer]:
            self.keep(layer, snapkv_select(weights, self.c.group, self.c.budget, self.window, self.kernel))
            self.done[layer] = True


# %% [attention] Attention over the kept rows
def rope(x, positions, theta):
    """A key is rotated once, by its own position, when it is written; evicting rows around it never
    changes that, so a kept key still says where it came from."""
    d = x.shape[-1]
    frequencies = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device).float() / d)
    angles = positions[:, None].float() * frequencies
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def forward(self, x, positions, cache, layer):
        c, t = self.c, x.shape[0]
        q = self.q(x).view(t, c.n_heads, c.head_dim).transpose(0, 1)
        k = self.k(x).view(t, c.n_kv_heads, c.head_dim).transpose(0, 1)
        v = self.v(x).view(t, c.n_kv_heads, c.head_dim).transpose(0, 1)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        k, v, kept = cache.append(layer, k, v, positions)  # kept: (n_kv_heads, rows) original positions
        k, v = k.repeat_interleave(c.group, dim=0), v.repeat_interleave(c.group, dim=0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)
        visible = kept.repeat_interleave(c.group, dim=0)[:, None, :] <= positions[None, :, None]
        weights = scores.masked_fill(~visible, float("-inf")).softmax(dim=-1)
        output = self.out((weights @ v).transpose(0, 1).reshape(t, -1))
        cache.observe(layer, weights)  # the policy may evict now, after this step has read its rows
        return output, weights


# %% [block] Transformer block
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


class MLP(nn.Module):
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


# %% [model] Decoding with an evicting cache
class Qwen2(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, ids, positions, cache):
        x = self.embedding(ids)
        for layer, block in enumerate(self.blocks):
            x = block(x, positions, cache, layer)
        return self.lm_head(self.norm(x))


@torch.no_grad()
def generate(model, prompt, cache, max_new_tokens=24, eos=151645):
    """Greedy decoding; a new token's position is the count of tokens written, not of rows kept."""
    logits = model(prompt, torch.arange(len(prompt)), cache)[-1]
    tokens = []
    while len(tokens) < max_new_tokens:
        token = logits.argmax()
        tokens.append(int(token))
        if token == eos or len(tokens) == max_new_tokens:
            break
        logits = model(token[None], torch.tensor([len(prompt) + len(tokens) - 1]), cache)[-1]
    return tokens


# tokens = generate(Qwen2(), prompt_ids, H2OCache(Config()))      # 16 heavy hitters + 16 newest
# tokens = generate(Qwen2(), prompt_ids, SnapKVCache(Config()))   # window 8, average pool 5
`;export{e as default};
