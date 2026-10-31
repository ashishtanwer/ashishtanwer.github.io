const e=`# MiMo-7B-RL, the model in the MTP tab, written as plain PyTorch blocks: a Qwen2-style decoder plus
# one multi-token-prediction (MTP) layer that guesses the token after next.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 151680
    d_model: int = 4096
    n_layers: int = 36
    n_heads: int = 32  # 128 dimensions per head
    n_kv_heads: int = 8  # grouped-query attention: four query heads share each key/value head
    d_mlp: int = 11008
    rope_theta: float = 640_000.0
    eps: float = 1e-5
    n_mtp_layers: int = 1  # num_nextn_predict_layers


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
def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    d = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = positions[:, None].float() * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Masked self-attention
class Attention(nn.Module):
    """Grouped-query attention; Q, K and V have a bias, the output projection has none."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * head_dim)
        self.out = nn.Linear(c.n_heads * head_dim, c.d_model, bias=False)

    def forward(self, x, positions):  # x: (batch, seq, d_model); positions: (seq,)
        c, (b, t, _) = self.c, x.shape
        q = self.q(x).view(b, t, c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = self.k(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [mlp] MLP
class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 4096 → 11008 → 4096."""

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

    def forward(self, x, positions):
        x = x + self.attention(self.norm1(x), positions)
        return x + self.mlp(self.norm2(x))


# %% [output] Next token
class Output(nn.Module):
    """Final RMSNorm, then the vocabulary projection (not tied to the embedding)."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, h):  # h: the last block's output
        return self.lm_head(self.norm(h))  # logits: (batch, seq, vocab)


# %% [mtp] MTP layer
class MTP(nn.Module):
    """Predicts token t+2 at position t from the main model's last block output h_t (before the final
    norm, which only the logits use) and the embedding of token t+1. Both are normalized, concatenated and
    projected back to 4096, then pass one ordinary transformer block (with its own attention over the earlier
    positions) and a final norm. The main model's lm_head turns the result into logits for t+2."""

    def __init__(self, c: Config):
        super().__init__()
        self.hidden_norm = RMSNorm(c.d_model, c.eps)
        self.token_norm = RMSNorm(c.d_model, c.eps)
        self.input_proj = nn.Linear(2 * c.d_model, c.d_model, bias=False)
        self.block = Block(c)
        self.final_norm = RMSNorm(c.d_model, c.eps)

    def forward(self, h, next_embedding, positions):
        # h: (batch, seq, d) the main model's state at t; next_embedding: E[token t+1] at the same rows
        x = self.input_proj(torch.cat((self.hidden_norm(h), self.token_norm(next_embedding)), dim=-1))
        return self.final_norm(self.block(x, positions))


# %% [model] MiMo with MTP
class MiMo(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.mtp = MTP(c)

    def forward(self, ids):
        """Logits for t+1 (main model) and for t+2 (MTP) at every position t. The MTP row at t needs
        token t+1, so the last position has no MTP prediction until the next token is known."""
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, positions)
        future = self.embedding(ids[:, 1:])
        future[:, positions[:-1] == 0] = 0  # vLLM zeroes the token embedding at position 0
        mtp_logits = self.output.lm_head(self.mtp(x[:, :-1], future, positions[:-1]))
        return self.output(x), mtp_logits


# %% [speculative] MTP as a one-token drafter
@torch.no_grad()
def mtp_decode(model, ids, max_new_tokens=64, eos=None):
    """Greedy decoding with the MTP layer as the draft model. Each step the main model scores the next
    token and the MTP layer's guess for the one after in one pass of two positions; the guess is kept if
    it equals the main model's own argmax, which then also gives the token after it for free. So every
    main-model pass emits 1 or 2 tokens, and the output is exactly greedy decoding's.
    (Written for clarity: it recomputes the whole sequence each step instead of keeping KV caches.)"""
    out = ids
    steps = []
    draft = None
    while out.shape[1] - ids.shape[1] < max_new_tokens:
        tokens = out if draft is None else torch.cat((out, draft.view(1, 1)), dim=1)
        logits, mtp_logits = model(tokens)
        if draft is None:
            emitted = logits[:, -1].argmax(dim=-1, keepdim=True)
        else:
            best = logits[:, -2:].argmax(dim=-1)  # the main model's choice at the draft's position and after
            emitted = best if best[0, 0] == draft else best[:, :1]
        steps.append(emitted.shape[1])
        out = torch.cat((out, emitted), dim=1)
        if eos is not None and (emitted == eos).any():
            break
        # The MTP layer's guess for the token after the last emitted one, from the state before it.
        _, mtp_logits = model(out)
        draft = mtp_logits[0, -1].argmax()
    return out, steps
`;export{e as default};
