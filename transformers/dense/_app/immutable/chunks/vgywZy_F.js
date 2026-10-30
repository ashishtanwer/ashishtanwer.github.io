const e=`# Speculative decoding with Qwen2.5, the models in the Speculative tab, written as plain PyTorch blocks.
# Qwen2.5-0.5B-Instruct drafts k = 4 tokens; Qwen2.5-1.5B-Instruct checks them in one pass.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Qwen2.5-1.5B-Instruct, the target
    vocab_size: int = 151936  # the draft and the target share one tokenizer
    d_model: int = 1536
    n_layers: int = 28
    n_heads: int = 12  # 128 dimensions per head
    n_kv_heads: int = 2  # grouped-query attention: six query heads share each key/value head
    d_mlp: int = 8960
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6


TARGET = Config()
DRAFT = Config(d_model=896, n_layers=24, n_heads=14, n_kv_heads=2, d_mlp=4864)  # Qwen2.5-0.5B-Instruct
K = 4  # drafted tokens per round


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token lookup. Positions enter later, through RoPE inside attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids)  # (batch, seq, d_model)


# %% [rope] RoPE
def rope(x, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    seq, d = x.shape[-2], x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(seq, device=x.device)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Masked self-attention
class Attention(nn.Module):
    """Grouped-query attention; Qwen2 gives Q, K and V a bias."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * head_dim)
        self.out = nn.Linear(c.n_heads * head_dim, c.d_model, bias=False)

    def forward(self, x):  # x: (batch, seq, d_model)
        c, (b, t, _) = self.c, x.shape
        q = self.q(x).view(b, t, c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = self.k(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [mlp] MLP
class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 1536 → 8960 → 1536 in the target."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(RMSNorm(x)), then x + MLP(RMSNorm(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [output] Next token
class Output(nn.Module):
    """Final RMSNorm, then the vocabulary projection, which shares the token embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(x))  # logits: (batch, seq, vocab)


def probabilities(logits, temperature=1.0):
    """Softmax over the vocabulary; the sampled run uses temperature 1."""
    return (logits / temperature).softmax(dim=-1)


# %% [qwen2] Qwen2.5 model
class Qwen2(nn.Module):
    """The same code for both models: Qwen2(DRAFT) and Qwen2(TARGET)."""

    def __init__(self, c: Config = TARGET):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# %% [draft-verify] Acceptance rule
def accept_greedy(drafts, target_logits):
    """Greedy: keep drafts while each equals the target's argmax at its position. The first mismatch
    is replaced by the target's argmax (a correction); if all k match, the target's argmax after
    them comes free (a bonus). The output is exactly what the target alone would decode greedily.

    drafts: (k,) token ids. target_logits: (k + 1, vocab), one row per drafted position and one after."""
    best = target_logits.argmax(dim=-1)  # (k + 1,)
    kept = int((drafts == best[:-1]).long().cumprod(dim=0).sum())  # drafts before the first mismatch
    return torch.cat((drafts[:kept], best[kept : kept + 1]))


def acceptance(p, q):
    """Chance of keeping a draft d: min(1, p(d) / q(d)), with p the target's and q the draft's probability."""
    return (p / q).clamp(max=1)


def residual(p, q):
    """Where the target wants more than the draft offered: max(0, p − q), renormalized."""
    r = (p - q).clamp(min=0)
    return r / r.sum(dim=-1, keepdim=True)


def accept_sampled(drafts, q, p):
    """Sampled: keep draft d when a uniform u < min(1, p(d) / q(d)). The first rejected position is
    resampled from residual(p, q) and the later drafts are dropped; if all k are kept, a bonus token is
    sampled from p after them. Every emitted token is distributed exactly as if the target had
    sampled it alone (Leviathan et al. 2023, Theorem 1).

    drafts: (k,) token ids, each sampled from its row of q: (k, vocab), the draft's probabilities.
    p: (k + 1, vocab), the target's probabilities at the same positions and one after."""
    k = len(drafts)
    rows = torch.arange(k)
    u = torch.rand(k)
    kept = int((u < acceptance(p[rows, drafts], q[rows, drafts])).long().cumprod(dim=0).sum())
    last = residual(p[kept], q[kept]) if kept < k else p[k]
    return torch.cat((drafts[:kept], torch.multinomial(last, 1)))


# %% [model] Speculative decoding
@torch.no_grad()
def speculative_decode(draft, target, ids, k=K, max_new_tokens=32, greedy=True, eos=None):
    """Each round the draft proposes k tokens, one forward pass each. The target then scores all of
    them in one pass (k + 1 positions), and the acceptance rule keeps a prefix of the drafts plus one
    token of the target's own, so every round emits between 1 and k + 1 tokens.

    ids: (seq,) the prompt. Without a KV cache each pass recomputes the whole prefix; the recording
    kept both models' caches and cropped the rejected drafts from them, which gives the same tokens."""
    prompt = len(ids)
    while len(ids) - prompt < max_new_tokens:
        proposal, q = ids, []
        for _ in range(k):  # draft: k small passes
            dist = probabilities(draft(proposal[None])[0, -1])
            token = dist.argmax() if greedy else torch.multinomial(dist, 1)[0]
            proposal, q = torch.cat((proposal, token[None])), q + [dist]
        drafts = proposal[len(ids) :]
        logits = target(proposal[None])[0, len(ids) - 1 :]  # verify: one big pass, k + 1 positions
        if greedy:
            new = accept_greedy(drafts, logits)
        else:
            new = accept_sampled(drafts, torch.stack(q), probabilities(logits))
        ids = torch.cat((ids, new))
        if eos is not None and eos in new:  # stop right after the end-of-answer token
            ids = ids[: len(ids) - len(new) + new.tolist().index(eos) + 1]
            break
    return ids[prompt:][:max_new_tokens]


# tokens = speculative_decode(Qwen2(DRAFT), Qwen2(TARGET), prompt_ids, eos=151645)  # <|im_end|>
`;export{e as default};
