const e=`# DFlash drafting for Qwen3-4B, the models in the Efficiency Explainer's DFlash tab, written as plain PyTorch blocks.
# z-lab/Qwen3-4B-DFlash-b16 is a five-layer block-diffusion drafter: it fills a block of 16 positions (the last
# accepted token, then 15 mask tokens) in one forward pass, reading Qwen/Qwen3-4B's context only through the outputs
# of five of its decoder blocks. Qwen3-4B checks the whole block in one forward and keeps the longest prefix that
# equals its own greedy choice, so the answer is exactly Qwen3-4B's greedy answer.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Qwen3-4B; the drafter's layers have the same width, heads and MLP
    vocab_size: int = 151936
    d_model: int = 2560
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8  # grouped-query attention: 4 query heads share each key/value head
    head_dim: int = 128
    d_mlp: int = 9728
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    draft_layers: int = 5
    feature_layers: tuple = (1, 9, 17, 25, 33)  # Qwen3 blocks (0-based) whose outputs the drafter reads
    block: int = 16  # the anchor + 15 drafts
    mask: int = 151669  # an embedding row Qwen3's tokenizer never emits


# %% [target] Qwen3-4B
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):  # in float32, then back to x's dtype, as Qwen3 does
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


def rope_tables(positions, c: Config, dtype):
    """cos, sin of position × theta^(-2i/d) for every pair (i, i + d/2) of a head vector. positions: (n,)."""
    inverse = 1.0 / c.rope_theta ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim)  # on the CPU, as Qwen3
    angles = positions.float()[:, None] * inverse.to(positions.device)[None]  # (n, d/2)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate(x, cos, sin):  # x: (heads, n, d); cos, sin: (n, d)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 2560 → 9728 → 2560."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Qwen3's causal attention with a KV cache: 32 query heads, 8 key/value heads, RMSNorm on every head's query
    and key before RoPE."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.eps)
        self.k_norm = RMSNorm(c.head_dim, c.eps)

    def forward(self, x, cos, sin, cache):
        """x: (n, d_model) new tokens; cache: [keys, values] of earlier tokens, each (kv_heads, L, head_dim),
        extended in place. Each new token sees the cache and the new tokens up to itself."""
        c, n = self.c, x.shape[0]
        q = rotate(self.q_norm(self.q(x).view(n, c.n_heads, c.head_dim)).transpose(0, 1), cos, sin)
        k = rotate(self.k_norm(self.k(x).view(n, c.n_kv_heads, c.head_dim)).transpose(0, 1), cos, sin)
        v = self.v(x).view(n, c.n_kv_heads, c.head_dim).transpose(0, 1)
        cache[0] = k if cache[0] is None else torch.cat((cache[0], k), dim=1)
        cache[1] = v if cache[1] is None else torch.cat((cache[1], v), dim=1)
        past = cache[0].shape[1] - n
        mask = torch.ones(n, past + n, dtype=torch.bool, device=x.device).tril(past)
        o = F.scaled_dot_product_attention(q[None], cache[0][None], cache[1][None], attn_mask=mask, enable_gqa=True)
        return self.out(o[0].transpose(0, 1).reshape(n, -1))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, cos, sin, cache):
        x = x + self.attention(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


class Qwen3(nn.Module):
    """The target. Returns the logits of the new tokens and the drafter's context features: the raw outputs of
    the blocks in c.feature_layers, side by side (n, 5 × d_model). Its LM head is its embedding (tied)."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)

    def new_cache(self):
        return [[None, None] for _ in self.blocks]

    def crop(self, cache, length):  # drop the keys and values of rejected drafts
        for layer in cache:
            layer[0], layer[1] = layer[0][:, :length], layer[1][:, :length]

    def lm_head(self, h):
        return F.linear(h, self.embedding.weight)

    def forward(self, ids, start, cache):
        x = self.embedding(ids)
        cos, sin = rope_tables(torch.arange(start, start + len(ids), device=ids.device), self.c, x.dtype)
        taps = []
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache[i])
            if i in self.c.feature_layers:
                taps.append(x)
        return self.lm_head(self.norm(x)).float(), torch.cat(taps, dim=-1)


# %% [features] Fusing Qwen3's features
class Fuse(nn.Module):
    """fc concatenates the five block outputs of a context token and maps them to one feature c (12 800 → 2560,
    no bias); hidden_norm normalizes it. Every draft layer reads the context only through these features."""

    def __init__(self, c: Config):
        super().__init__()
        self.fc = nn.Linear(len(c.feature_layers) * c.d_model, c.d_model, bias=False)
        self.hidden_norm = RMSNorm(c.d_model, c.eps)

    def forward(self, taps):  # (n, 5 × d_model) → (n, d_model)
        return self.hidden_norm(self.fc(taps))


# %% [attention] Draft attention over the context and the block
class DraftAttention(nn.Module):
    """Qwen3 attention whose keys and values come from two places: this layer's own projections of the context
    features c (cached, one entry per context position, never recomputed) and of the block's hidden states. The
    block's queries see everything: no causal mask, so every slot reads every other slot."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)
        self.q_norm = RMSNorm(c.head_dim, c.eps)
        self.k_norm = RMSNorm(c.head_dim, c.eps)

    def forward(self, x, context, cos, sin, cache):
        """x: (b, d_model) block; context: (m, d_model) features of the m context positions not yet cached;
        cos, sin: (m + b, head_dim) for those positions then the block's; cache: [keys, values] (kv_heads, L, d),
        to which the context's entries are appended (the block's are used once and dropped)."""
        c, b, m = self.c, x.shape[0], context.shape[0]
        q = rotate(self.q_norm(self.q(x).view(b, c.n_heads, c.head_dim)).transpose(0, 1), cos[m:], sin[m:])
        k = torch.cat((self.k(context), self.k(x))).view(m + b, c.n_kv_heads, c.head_dim)
        k = rotate(self.k_norm(k).transpose(0, 1), cos, sin)
        v = torch.cat((self.v(context), self.v(x))).view(m + b, c.n_kv_heads, c.head_dim).transpose(0, 1)
        if cache[0] is not None:
            k, v = torch.cat((cache[0], k), dim=1), torch.cat((cache[1], v), dim=1)
        cache[0], cache[1] = k[:, :-b], v[:, :-b]
        o = F.scaled_dot_product_attention(q[None], k[None], v[None], scale=c.head_dim**-0.5, enable_gqa=True)
        return self.out(o[0].transpose(0, 1).reshape(b, -1))


# %% [block] Draft layer
class DraftLayer(nn.Module):
    """A Qwen3 decoder layer whose attention also reads the context features (pre-norm, residual)."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = DraftAttention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, context, cos, sin, cache):
        x = x + self.attention(self.norm1(x), context, cos, sin, cache)
        return x + self.mlp(self.norm2(x))


# %% [draft] The DFlash drafter: one denoising pass
class DFlash(nn.Module):
    """Five draft layers over a block of 16 embeddings: the anchor (the last accepted token, which Qwen3 has not
    run yet) and 15 copies of the mask token, embedded by Qwen3's own embedding. One pass fills all 15 mask slots;
    Qwen3's LM head turns them into drafts."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.fuse = Fuse(c)
        self.layers = nn.ModuleList(DraftLayer(c) for _ in range(c.draft_layers))
        self.norm = RMSNorm(c.d_model, c.eps)

    def new_cache(self):
        return [[None, None] for _ in self.layers]

    def forward(self, block, taps, start, cache):
        """block: (b, d_model) embeddings at positions start …; taps: (m, 5 × d_model) Qwen3 features of the
        positions start - m … start - 1. Returns the block's hidden states (b, d_model)."""
        context = self.fuse(taps)
        m = context.shape[0]
        cos, sin = rope_tables(torch.arange(start - m, start + len(block), device=block.device), self.c, block.dtype)
        x = block
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, context, cos, sin, layer_cache)
        return self.norm(x)

    @torch.no_grad()
    def propose(self, target, anchor, taps, start, cache):
        """The 15 drafts after \`anchor\` and the drafter's logits for them."""
        ids = torch.full((self.c.block,), self.c.mask, device=taps.device)
        ids[0] = anchor
        hidden = self(target.embedding(ids), taps, start, cache)
        logits = target.lm_head(hidden[1:])
        return logits.argmax(-1), logits


# %% [verify] Greedy verification
def verify(drafts, logits):
    """drafts: (k,) the tokens after the anchor; logits: (k + 1, vocab) Qwen3's prediction after the anchor and
    after each draft, from one forward. Keeps drafts while they equal Qwen3's argmax; the argmax at the first
    difference (or after the last draft) is Qwen3's own next token. Returns (kept, next_token)."""
    best = logits.argmax(-1)
    kept = 0
    while kept < len(drafts) and drafts[kept] == best[kept]:
        kept += 1
    return kept, int(best[kept])


# %% [model] DFlash decoding
@torch.no_grad()
def generate(target, drafter, prompt, max_new_tokens=320, eos=(151645, 151643)):
    """Greedy speculative decoding: Qwen3 reads the prompt once; then each step the drafter proposes 15 tokens and
    Qwen3 checks them in one forward. Returns the answer and the drafts kept per step. \`target\` is any Qwen3-4B
    with forward(ids, start, cache) → (logits, taps), new_cache(), crop(cache, length), embedding and lm_head."""
    t_cache, d_cache = target.new_cache(), drafter.new_cache()
    logits, taps = target(prompt, 0, t_cache)
    answer, start, kept_per_step = [int(logits[-1].argmax())], len(prompt), []
    while not set(answer) & set(eos) and len(answer) < max_new_tokens:
        drafts, _ = drafter.propose(target, answer[-1], taps, start, d_cache)
        block = torch.cat((torch.tensor([answer[-1]], device=drafts.device), drafts))
        logits, block_taps = target(block, start, t_cache)
        kept, token = verify(drafts.tolist(), logits)
        target.crop(t_cache, start + kept + 1)
        taps = block_taps[: kept + 1]  # the anchor's and kept drafts' features: next step's new context
        start += kept + 1
        answer += drafts[:kept].tolist() + [token]
        kept_per_step.append(kept)
    cut = next((i + 1 for i, t in enumerate(answer) if t in eos), len(answer))
    return answer[: min(cut, max_new_tokens)], kept_per_step
`;export{e as default};
