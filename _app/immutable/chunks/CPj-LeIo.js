const e=`# Lookahead decoding on Vicuna-7B v1.3, the Efficiency Explainer's Lookahead tab, written as plain PyTorch blocks.
# No draft model and no extra weights: one lmsys/vicuna-7b-v1.3 forward per step runs a 2-D window of Jacobi
# guesses (the lookahead branch) and the pool's n-grams that start with the last token (the verification branch).
# Each "# %% [id]" section is one block of the diagram.
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Vicuna-7B v1.3: LLaMA-7B's shape; lookahead settings for a 7B model
    vocab_size: int = 32000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # 128 dimensions per head
    d_mlp: int = 11008
    rope_theta: float = 10000.0
    eps: float = 1e-6
    level: int = 5  # N: n-gram length; the window has N - 1 levels
    window: int = 7  # W: columns of the window
    guess_set: int = 7  # G: n-grams kept per first token


# %% [block] Vicuna block
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):  # in float32, as Llama does, then back to x's dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d). positions: (seq,) — tree
    tokens share positions: a node sits at prefix length + its depth."""
    d = x.shape[-1]
    frequencies = (1.0 / theta ** (torch.arange(0, d, 2).float() / d)).to(x.device)  # on the CPU, as Llama does
    angles = positions.float()[:, None] * frequencies  # (seq, d/2)
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Attention(nn.Module):
    """Multi-head attention over the cached keys and values plus the new tokens, under any boolean mask."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, positions, mask, cache=None):
        """x: (seq, d_model). mask: (seq, cached + seq), True = may attend. Returns the output and the new (k, v)."""
        c, t = self.c, x.shape[0]
        q = self.q(x).view(1, t, c.n_heads, -1).transpose(1, 2)  # (1, heads, seq, head_dim), as Llama keeps it
        k = self.k(x).view(1, t, c.n_heads, -1).transpose(1, 2)
        v = self.v(x).view(1, t, c.n_heads, -1).transpose(1, 2)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        keys = k if cache is None else torch.cat((cache[0], k), dim=2)
        values = v if cache is None else torch.cat((cache[1], v), dim=2)
        o = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(t, -1)), (k, v)


class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 4096 → 11008 → 4096."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm residual block: x + Attention(RMSNorm(x)), then x + MLP(RMSNorm(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, positions, mask, cache=None):
        a, new = self.attention(self.norm1(x), positions, mask, cache)
        x = x + a
        return x + self.mlp(self.norm2(x)), new


# %% [window] Lookahead window
def start_window(prompt, c: Config, seed=0):
    """Level 0 to begin with: W + N - 3 tokens copied at random from the prompt. The other levels fill in."""
    rng = random.Random(seed)
    return [[rng.choice(prompt) for _ in range(c.window + c.level - 3)]]


def collect(levels, last, outputs, c: Config):
    """Every column read downwards, finished by its new output: N tokens at consecutive positions. Column 0
    starts with x_p (level 0 holds columns 1 …)."""
    return [(last if i == 0 else levels[0][i - 1], tuple(levels[m][i] for m in range(1, c.level - 1)) + (outputs[i],))
            for i in range(len(outputs))]


def advance(levels, outputs, c: Config):
    """One Jacobi iteration after step 1. Full window: every level moves up one (level 0 loses its first column)
    and the new outputs become the top level. While filling: every level drops a column and the outputs add a
    level."""
    if len(levels) == c.level - 1:
        return [levels[1][1:]] + [list(level) for level in levels[2:]] + [list(outputs)]
    return [level[1:] for level in levels] + [list(outputs[1:])]


# %% [mask] One forward, one mask
class Vicuna(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, ids, positions, mask, cache=None):
        """ids, positions: (seq,). mask: (seq, seq) over the new tokens; every new token also sees the whole cache.
        A causal mask makes this a prefill or a greedy step; a tree mask scores every path of a candidate tree at
        once. Returns the final-norm hidden states (what the heads read), the logits and the new keys and values."""
        cached = 0 if cache is None else cache[0][0].shape[2]
        full = torch.cat((torch.ones(len(ids), cached, dtype=torch.bool, device=ids.device), mask), dim=1)
        x, new = self.embedding(ids), []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, positions, full, None if cache is None else cache[i])
            new.append(kv)
        h = self.norm(x)
        return h, self.lm_head(h), new


def extend(cache, new, keep=None):
    """Append the new tokens' keys and values to the cache: all of them, or only the tree nodes in \`keep\`."""
    if keep is not None:
        new = [(k[:, :, keep], v[:, :, keep]) for k, v in new]
    if cache is None:
        return new
    return [(torch.cat((ck, k), dim=2), torch.cat((cv, v), dim=2)) for (ck, cv), (k, v) in zip(cache, new)]


def layout(levels, n_guesses, c: Config):
    """The step's input after the cache: x_p, the window (level 0 from column 1, then levels 1 …), then the
    candidate n-grams. Returns each token's position offset from p and the (n, n) mask: a window token at
    (level l, column i) sees level 0's columns 0 … i and its own column's levels 1 … l; a candidate's token sees
    x_p and the candidate's earlier tokens; x_p sees itself. Everything sees the cache."""
    columns = len(levels[0]) + 1
    cells = [(0, i) for i in range(columns)] + [(l, i) for l in range(1, len(levels)) for i in range(columns)]
    index = {cell: k for k, cell in enumerate(cells)}
    n = len(cells) + n_guesses * (c.level - 1)
    mask = torch.zeros(n, n, dtype=torch.bool)
    offsets = []
    for row, (level, column) in enumerate(cells):
        offsets.append(level + column)
        for i in range(column + 1):
            mask[row, index[(0, i)]] = True
        for m in range(1, level + 1):
            mask[row, index[(m, column)]] = True
    for k in range(n_guesses):
        for j in range(c.level - 1):
            row = len(cells) + k * (c.level - 1) + j
            offsets.append(1 + j)
            mask[row, 0] = True
            mask[row, row - j : row + 1] = True
    return torch.tensor(offsets), mask


# %% [pool] N-gram pool
def remember(pool, key, gram, c: Config):
    """At most G n-grams per first token; a repeat moves to the back, the oldest drops out when full."""
    grams = pool.setdefault(key, [])
    if gram in grams:
        grams.remove(gram)
    elif len(grams) == c.guess_set:
        grams.pop(0)
    grams.append(gram)


# %% [verify] Verification branch
def verify(candidates, first, guess_argmax):
    """first: Vicuna's argmax after x_p; guess_argmax[k][j]: its argmax after candidate k's token j. A candidate
    is kept while each token equals Vicuna's argmax before it; the longest wins (the first on ties) and Vicuna's
    next token after it is kept too. Returns the tokens to append, the winner and every candidate's kept count."""
    best, best_kept, matched = -1, -1, []
    for k, gram in enumerate(candidates):
        correct = [first] + list(guess_argmax[k])
        m = 0
        while m < len(gram) and gram[m] == correct[m]:
            m += 1
        matched.append(m)
        if m > best_kept:
            best, best_kept = k, m
    if best < 0:
        return [first], best, matched
    return ([first] + list(guess_argmax[best]))[: best_kept + 1], best, matched


# %% [model] Lookahead decoding
@torch.no_grad()
def lookahead_generate(vicuna, prompt, max_new_tokens, eos=2, seed=0):
    """Greedy decoding with lookahead: one Vicuna forward per step, appending 1 … N tokens."""
    c, n = vicuna.c, len(prompt)
    levels = start_window(prompt.tolist(), c, seed)
    ids = torch.cat((prompt, torch.tensor(levels[0])))
    _, logits, new = vicuna(ids, torch.arange(len(ids)), torch.ones(len(ids), len(ids), dtype=torch.bool).tril())
    cache = [(k[:, :, :n], v[:, :, :n]) for k, v in new]  # the window's keys and values are not kept
    argmax = logits.argmax(dim=-1)
    out, pool = [int(argmax[n - 1])], {}
    levels = [levels[0][1:], argmax[n:].tolist()]  # level 1: the next guess after every level-0 token
    p = n
    while len(out) < max_new_tokens and eos not in out:
        last, full = out[-1], len(levels) == c.level - 1
        candidates = list(pool.get(last, [])) if full else []
        offsets, mask = layout(levels, len(candidates), c)
        ids = [last] + levels[0] + [t for level in levels[1:] for t in level] + [t for g in candidates for t in g]
        _, logits, new = vicuna(torch.tensor(ids), p + offsets, mask, cache)
        argmax = logits.argmax(dim=-1).tolist()
        columns, top = len(levels[0]) + 1, (len(levels) - 1) * (len(levels[0]) + 1)
        outputs = argmax[top : top + columns]
        guesses = [argmax[top + columns + k * (c.level - 1) : top + columns + (k + 1) * (c.level - 1)]
                   for k in range(len(candidates))]
        emitted, best, _ = verify(candidates, argmax[0], guesses)
        keep = [0] + [top + columns + best * (c.level - 1) + j for j in range(len(emitted) - 1)]
        cache = extend(cache, new, keep)  # x_p and the kept candidate tokens
        if full:
            for key, gram in collect(levels, last, outputs, c):
                remember(pool, key, gram, c)
        levels = advance(levels, outputs, c)
        out += emitted
        p += len(emitted)
    return out[: out.index(eos) + 1] if eos in out else out


@torch.no_grad()
def greedy_generate(vicuna, prompt, max_new_tokens, eos=2):
    """Plain greedy decoding, one Vicuna forward per token: what Medusa must reproduce."""
    n = len(prompt)
    _, logits, cache = vicuna(prompt, torch.arange(n), torch.ones(n, n, dtype=torch.bool).tril())
    out = []
    while len(out) < max_new_tokens:
        out.append(int(logits[-1].argmax()))
        if out[-1] == eos:
            break
        position = torch.tensor([n + len(out) - 1])
        _, logits, new = vicuna(torch.tensor(out[-1:]), position, torch.ones(1, 1, dtype=torch.bool), cache)
        cache = extend(cache, new)
    return out
`;export{e as default};
