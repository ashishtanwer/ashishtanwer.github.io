const e=`# DSpark drafting for Qwen3-4B, the models in the Efficiency Explainer's DSpark tab, written as plain PyTorch blocks.
# deepseek-ai/dspark_qwen3_4b_block7 keeps DFlash's parallel backbone (five Qwen3 layers that read Qwen/Qwen3-4B's
# context through features from five of its blocks) over a block of 7 positions, adds a sequential stage (a rank-256
# "Markov" bias from the token drafted just before each slot) and a confidence head, and verifies only as many drafts
# as a scheduler expects to pay off (arXiv 2607.05147, Algorithm 1). Qwen3-4B checks them in one forward; the
# answer is exactly its greedy answer. The target is the Qwen3 of efficiency_dflash.py.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # the drafter's layers have Qwen3-4B's width, heads and MLP
    vocab_size: int = 151936
    d_model: int = 2560
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_mlp: int = 9728
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    draft_layers: int = 5
    feature_layers: tuple = (1, 9, 17, 25, 33)  # Qwen3 blocks (0-based) whose outputs the drafter reads
    block: int = 7  # the anchor + 6 masks in, 7 drafts out
    mask: int = 151669
    markov_rank: int = 256


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
    inverse = 1.0 / c.rope_theta ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim)  # on the CPU, as Qwen3
    angles = positions.float()[:, None] * inverse.to(positions.device)[None]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [backbone] Parallel backbone
class DraftAttention(nn.Module):
    """Attention whose keys and values are this layer's projections of the context features (cached) and of the
    block; no causal mask (DFlash's design, unchanged)."""

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
        c, b, m = self.c, x.shape[0], context.shape[0]
        q = rotate(self.q_norm(self.q(x).view(b, c.n_heads, c.head_dim)).transpose(0, 1), cos[m:], sin[m:])
        k = torch.cat((self.k(context), self.k(x))).view(m + b, c.n_kv_heads, c.head_dim)
        k = rotate(self.k_norm(k).transpose(0, 1), cos, sin)
        v = torch.cat((self.v(context), self.v(x))).view(m + b, c.n_kv_heads, c.head_dim).transpose(0, 1)
        if cache[0] is not None:
            k, v = torch.cat((cache[0], k), dim=1), torch.cat((cache[1], v), dim=1)
        cache[0], cache[1] = k[:, :-b], v[:, :-b]  # keep the context's entries, drop the block's
        o = F.scaled_dot_product_attention(q[None], k[None], v[None], scale=c.head_dim**-0.5, enable_gqa=True)
        return self.out(o[0].transpose(0, 1).reshape(b, -1))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DraftLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = DraftAttention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, context, cos, sin, cache):
        x = x + self.attention(self.norm1(x), context, cos, sin, cache)
        return x + self.mlp(self.norm2(x))


class Backbone(nn.Module):
    """fc + hidden_norm fuse Qwen3's five block outputs per context token; five draft layers run once over the
    block (the anchor and 6 masks). Unlike DFlash every slot predicts, the anchor's included: slot k's hidden
    state h_k gives the base logits U_k for the token k + 1 places after the anchor."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.fc = nn.Linear(len(c.feature_layers) * c.d_model, c.d_model, bias=False)
        self.hidden_norm = RMSNorm(c.d_model, c.eps)
        self.layers = nn.ModuleList(DraftLayer(c) for _ in range(c.draft_layers))
        self.norm = RMSNorm(c.d_model, c.eps)

    def new_cache(self):
        return [[None, None] for _ in self.layers]

    def forward(self, block, taps, start, cache):
        """block: (7, d_model) embeddings at positions start …; taps: (m, 5 × d_model) Qwen3 features of the
        positions start - m … start - 1. Returns h: (7, d_model)."""
        context = self.hidden_norm(self.fc(taps))
        m = context.shape[0]
        cos, sin = rope_tables(torch.arange(start - m, start + len(block), device=block.device), self.c, block.dtype)
        x = block
        for layer, layer_cache in zip(self.layers, cache):
            x = layer(x, context, cos, sin, layer_cache)
        return self.norm(x)


# %% [markov] Sequential stage: the Markov head
class Markov(nn.Module):
    """A low-rank bigram bias B(x_{k-1}, ·) = W2 · W1[x_{k-1}] added to slot k's base logits. Applied left to
    right, each slot's draft depends on the one picked before it, which the parallel backbone cannot see."""

    def __init__(self, c: Config):
        super().__init__()
        self.w1 = nn.Embedding(c.vocab_size, c.markov_rank)  # previous token → 256
        self.w2 = nn.Linear(c.markov_rank, c.vocab_size, bias=False)  # 256 → vocabulary

    def forward(self, base_logits, anchor):
        """base_logits: (7, vocab) U_k from the backbone; anchor: the token before slot 1. Returns the drafts (7,)
        and the corrected logits (7, vocab)."""
        drafts, logits, previous = [], [], torch.tensor([anchor], device=base_logits.device)
        for k in range(len(base_logits)):
            z = base_logits[k : k + 1] + self.w2(self.w1(previous))
            previous = z.argmax(-1)
            drafts.append(previous)
            logits.append(z)
        return torch.cat(drafts), torch.cat(logits)


# %% [confidence] Confidence head and prefix survival
class Confidence(nn.Module):
    """c_k = σ(w · [h_k; W1[x_{k-1}]] + b): the chance draft k survives verification if every draft before it did.
    Their running product a_k = c_1 ⋯ c_k is the chance the first k drafts all survive."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.d_model + c.markov_rank, 1)

    def forward(self, hidden, previous_embeddings):
        c = self.proj(torch.cat((hidden, previous_embeddings), dim=-1)).float()[:, 0].sigmoid()
        return c, c.cumprod(0)


# %% [schedule] Hardware-aware prefix scheduler
def schedule(survival, seconds):
    """Algorithm 1 for one request. seconds[B - 1]: the target's measured time for a forward of B tokens, so
    SPS(B) = 1 / seconds[B - 1]. Verifying ℓ drafts sends B = 1 + ℓ tokens and is expected to yield
    τ = 1 + a_1 + … + a_ℓ tokens; admit drafts in order while Θ = τ · SPS(B) grows and stop at the first that does
    not (looking further would let a draft's own value decide whether it is checked). Returns ℓ."""
    tau, best, chosen = 1.0, 1.0 / seconds[0], 0
    for length, a in enumerate(survival.tolist(), start=1):
        tau += a
        theta = tau / seconds[length]
        if theta <= best:
            break
        best, chosen = theta, length
    return chosen


# %% [verify] Greedy verification
def verify(drafts, logits):
    """drafts: (ℓ,) the scheduled drafts; logits: (ℓ + 1, vocab) Qwen3's predictions after the anchor and each
    draft. Keeps drafts while they equal Qwen3's argmax; returns (kept, Qwen3's next token)."""
    best = logits.argmax(-1)
    kept = 0
    while kept < len(drafts) and drafts[kept] == best[kept]:
        kept += 1
    return kept, int(best[kept])


# %% [model] DSpark decoding
class DSpark(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.backbone = Backbone(c)
        self.markov = Markov(c)
        self.confidence = Confidence(c)

    @torch.no_grad()
    def propose(self, target, anchor, taps, start, cache):
        """Drafts for the 7 tokens after \`anchor\`, their confidences and prefix survival."""
        ids = torch.full((self.c.block,), self.c.mask, device=taps.device)
        ids[0] = anchor
        hidden = self.backbone(target.embedding(ids), taps, start, cache)
        drafts, _ = self.markov(target.lm_head(hidden), anchor)
        previous = torch.cat((ids[:1], drafts[:-1]))
        confidence, survival = self.confidence(hidden, self.markov.w1(previous))
        return drafts, confidence, survival


@torch.no_grad()
def generate(target, drafter, prompt, seconds=None, max_new_tokens=320, eos=(151645, 151643)):
    """Greedy speculative decoding with scheduled verification (\`seconds\`: the measured cost table; None verifies
    all 7 drafts). \`target\` is a Qwen3-4B as in efficiency_dflash.py. Returns the answer and, per step, the drafts
    verified and kept."""
    t_cache, d_cache = target.new_cache(), drafter.backbone.new_cache()
    logits, taps = target(prompt, 0, t_cache)
    answer, start, record = [int(logits[-1].argmax())], len(prompt), []
    while not set(answer) & set(eos) and len(answer) < max_new_tokens:
        drafts, _, survival = drafter.propose(target, answer[-1], taps, start, d_cache)
        length = schedule(survival, seconds) if seconds is not None else len(drafts)
        block = torch.cat((torch.tensor([answer[-1]], device=drafts.device), drafts[:length]))
        logits, block_taps = target(block, start, t_cache)
        kept, token = verify(drafts[:length].tolist(), logits)
        target.crop(t_cache, start + kept + 1)
        taps = block_taps[: kept + 1]
        start += kept + 1
        answer += drafts[:kept].tolist() + [token]
        record.append((length, kept))
    cut = next((i + 1 for i, t in enumerate(answer) if t in eos), len(answer))
    return answer[: min(cut, max_new_tokens)], record
`;export{e as default};
