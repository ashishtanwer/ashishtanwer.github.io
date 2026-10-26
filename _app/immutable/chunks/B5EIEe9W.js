const e=`# Ouro-1.4B, the looped model in the Looped tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 49152
    d_model: int = 2048
    n_layers: int = 24  # one stack of 24 blocks ...
    n_loops: int = 4  # ... applied 4 times with the same weights
    n_heads: int = 16  # 128 dimensions per head, no grouped-query sharing
    d_mlp: int = 5632
    rope_theta: float = 1_000_000.0
    exit_threshold: float = 1.0  # the release config: stop once the exit probabilities add up to 1
    eps: float = 1e-6


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
    """Token lookup, used once: loops 2 to 4 start from the previous loop's output instead."""

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
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x):  # x: (batch, seq, d_model)
        c, (b, t, d) = self.c, x.shape
        # Split into heads: (batch, heads, seq, head_dim)
        q, k, v = (proj(x).view(b, t, c.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, d))


# %% [mlp] MLP
class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 2048 → 5632 → 2048."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# %% [block] Transformer block
class Block(nn.Module):
    """Sandwich norms: x + RMSNorm(Attention(RMSNorm(x))), then x + RMSNorm(MLP(RMSNorm(x)))."""

    def __init__(self, c: Config):
        super().__init__()
        self.attention_in = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.attention_out = RMSNorm(c.d_model, c.eps)
        self.mlp_in = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)
        self.mlp_out = RMSNorm(c.d_model, c.eps)

    def forward(self, x):
        x = x + self.attention_out(self.attention(self.attention_in(x)))
        return x + self.mlp_out(self.mlp(self.mlp_in(x)))


# %% [loop] Loop
class Loop(nn.Module):
    """The same 24 blocks run 4 times. After every pass the final RMSNorm is applied; that state is
    what the next pass starts from, what the exit gate reads and what the LM head can read."""

    def __init__(self, c: Config):
        super().__init__()
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.n_loops = c.n_loops

    def forward(self, x):  # x: token embeddings, (batch, seq, d_model)
        states = []
        for _ in range(self.n_loops):
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            states.append(x)
        return states  # one (batch, seq, d_model) state per loop


# %% [exit-gate] Exit gate
class ExitGate(nn.Module):
    """λ_r = σ(w · h_r + b) is the chance of stopping after loop r, given that loop r was reached.
    Stopping exactly after loop r then has probability λ_r ∏_{j<r} (1 − λ_j); the last loop takes
    whatever is left, so the exit probabilities add up to 1."""

    def __init__(self, c: Config):
        super().__init__()
        self.linear = nn.Linear(c.d_model, 1)  # 2,049 parameters, shared by every loop

    def forward(self, states):  # one (batch, seq, d_model) state per loop
        lambdas = [torch.sigmoid(self.linear(h)).squeeze(-1) for h in states]
        exits, remaining = [], torch.ones_like(lambdas[0])
        for r, lam in enumerate(lambdas):
            exits.append(remaining if r == len(lambdas) - 1 else lam * remaining)
            remaining = remaining * (1 - lam)
        return torch.stack(exits, dim=-1)  # (batch, seq, loops)


def exit_loop(exits, threshold):
    """Each position reads the first loop where the exit probabilities so far reach the threshold,
    or the last loop. All loops run either way; the gate only picks which state is read."""
    reached = exits.cumsum(dim=-1) >= threshold
    first = reached.float().argmax(dim=-1)
    return torch.where(reached.any(dim=-1), first, exits.shape[-1] - 1)  # (batch, seq) loop indices


# %% [output] Next token
class Output(nn.Module):
    """The LM head (not tied to the embedding) reads the state after whichever loop a position stops at."""

    def __init__(self, c: Config):
        super().__init__()
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, state):
        return self.lm_head(state)  # logits: (batch, seq, vocab)


def next_token_probabilities(logits, temperature=1.0, top_k=None, top_p=None):
    """The last position's logits, scaled by temperature and cut to top-k or top-p, as probabilities."""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        p = sorted_logits.softmax(dim=-1)
        drop = p.cumsum(dim=-1) - p >= top_p  # keep the smallest set whose mass reaches top_p
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits.softmax(dim=-1)


# %% [model] Full model
class Ouro(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.loop = Loop(c)
        self.gate = ExitGate(c)
        self.output = Output(c)
        self.exit_threshold = c.exit_threshold

    def forward(self, ids, exit_threshold=None):
        states = self.loop(self.embedding(ids))
        stop = exit_loop(self.gate(states), exit_threshold or self.exit_threshold)
        batch, seq = torch.meshgrid(torch.arange(ids.shape[0]), torch.arange(ids.shape[1]), indexing="ij")
        return self.output(torch.stack(states, dim=2)[batch, seq, stop])  # each position's chosen loop


# probabilities = next_token_probabilities(Ouro()(ids), temperature=0.8, top_k=5)
`;export{e as default};
