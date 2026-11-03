const e=`# From GPT-2 to gpt-oss one change at a time, as the Pretraining Explainer trains it: the shared small GPT (4 layers,
# d_model 256, 4 heads, context 256, tied head, TinyStories, AdamW) and six rungs, each adding one change to the one
# before: rotary positions instead of learned ones, RMSNorm instead of LayerNorm, a SwiGLU MLP instead of GELU,
# grouped-query attention, a mixture of experts instead of the dense MLP, and gpt-oss's attention (a learned sink per
# head, and a sliding window on every other layer). Every rung trains on the same batches with the same seed and the
# same number of tokens; scripts/pretraining/arch_ladder_runs.py trains with these blocks.
import math
import zlib
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    context: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    positions: str = "learned"  # "learned" | "rope"
    norm: str = "layernorm"  # "layernorm" | "rmsnorm"
    mlp: str = "gelu"  # "gelu" | "swiglu"
    kv_heads: int = 4  # 4 = multi-head attention; fewer = grouped-query attention
    experts: int = 0  # 0 = one dense MLP per layer
    top_k: int = 2
    sinks: bool = False  # a learned sink logit per head
    window: int = 0  # > 0: layers 0, 2, … attend to the last \`window\` tokens only
    rope_base: float = 10000.0
    aux_weight: float = 0.01  # the experts' load-balancing loss
    clip: float = 1.0  # gradient norm clipped to this, as GPT-3, Llama and later runs do
    eps: float = 1e-5


# %% [rungs] The ladder
RUNGS = [
    ("gpt2", "GPT-2", {}),
    ("rope", "RoPE", {"positions": "rope"}),
    ("rmsnorm", "RMSNorm", {"norm": "rmsnorm"}),
    ("swiglu", "SwiGLU", {"mlp": "swiglu"}),
    ("gqa", "GQA", {"kv_heads": 2}),
    ("moe", "MoE", {"experts": 8}),
    ("sinks-swa", "sinks + sliding window", {"sinks": True, "window": 64}),
]


def rung_config(index: int, base: Config = Config()) -> Config:
    """Rung \`index\` carries every change up to and including its own."""
    c = base
    for _, _, change in RUNGS[: index + 1]:
        c = replace(c, **change)
    return c


# %% [rope] Rotary positions
def rope(x, base):
    """Rotate each pair of features (i, i + head_dim/2) of a query or key at position p by the angle p·θᵢ, θᵢ =
    base^(−2i/head_dim): the dot product of a query and a key then depends on their distance, not their positions."""
    T, half = x.shape[-2], x.shape[-1] // 2
    theta = base ** (-torch.arange(half, dtype=torch.float32, device=x.device) / half)
    angle = torch.arange(T, dtype=torch.float32, device=x.device)[:, None] * theta  # (T, half)
    cos, sin = angle.cos(), angle.sin()
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


# %% [norm] LayerNorm and RMSNorm
class Norm(nn.Module):
    """LayerNorm centres each token's features and divides by their standard deviation (gain and bias); RMSNorm skips
    the centring and the bias and divides by the root mean square."""

    def __init__(self, c: Config):
        super().__init__()
        self.rms, self.eps = c.norm == "rmsnorm", c.eps
        self.gain = nn.Parameter(torch.ones(c.d_model))
        self.bias = None if self.rms else nn.Parameter(torch.zeros(c.d_model))

    def forward(self, x):
        if not self.rms:
            x = x - x.mean(-1, keepdim=True)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gain
        return x if self.rms else x + self.bias


# %% [attention] Attention: GQA, sinks and the sliding window
class Attention(nn.Module):
    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.c, self.head_dim = c, c.d_model // c.n_head
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.kv_heads * self.head_dim, bias=False)  # GQA: fewer key and value heads
        self.v = nn.Linear(c.d_model, c.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(c.d_model, c.d_model, bias=False)
        self.sink = nn.Parameter(torch.zeros(c.n_head)) if c.sinks else None
        self.window = c.window if c.window and layer % 2 == 0 else 0
        self.weights = None  # (batch, heads, T, T) and the sink's share (batch, heads, T), kept when asked

    def forward(self, x, keep=False):
        c, B, T = self.c, x.shape[0], x.shape[1]
        q = self.q(x).view(B, T, c.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, c.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, c.kv_heads, self.head_dim).transpose(1, 2)
        if c.positions == "rope":
            q, k = rope(q, c.rope_base), rope(k, c.rope_base)
        group = c.n_head // c.kv_heads  # query heads sharing each key/value head
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        i = torch.arange(T, device=x.device)
        hidden = i[None, :] > i[:, None]  # causal
        if self.window:
            hidden |= i[None, :] <= i[:, None] - self.window  # only the last \`window\` keys
        scores = scores.masked_fill(hidden, float("-inf"))
        if self.sink is not None:  # one more column whose logit is learned, softmaxed with the rest, then dropped
            scores = torch.cat([scores, self.sink.view(1, -1, 1, 1).expand(B, -1, T, 1)], dim=-1)
        weights = scores.softmax(-1)
        sink_share = weights[..., -1] if self.sink is not None else None
        weights = weights[..., :T]
        if keep:
            self.weights = (weights.detach(), None if sink_share is None else sink_share.detach())
        return self.o((weights @ v).transpose(1, 2).reshape(B, T, c.d_model))


# %% [mlp] GELU and SwiGLU MLPs
def swiglu_hidden(d_model):
    """SwiGLU has three matrices instead of two; a hidden size of 8/3·d_model (rounded to 8) keeps the parameters."""
    return 8 * round(8 * d_model / 3 / 8)


class MLP(nn.Module):
    def __init__(self, c: Config, hidden=None):
        super().__init__()
        self.swiglu = c.mlp == "swiglu"
        hidden = hidden or (swiglu_hidden(c.d_model) if self.swiglu else 4 * c.d_model)
        self.fc = nn.Linear(c.d_model, hidden, bias=False)
        self.gate = nn.Linear(c.d_model, hidden, bias=False) if self.swiglu else None
        self.proj = nn.Linear(hidden, c.d_model, bias=False)

    def forward(self, x):
        h = F.silu(self.gate(x)) * self.fc(x) if self.swiglu else F.gelu(self.fc(x))
        return self.proj(h)


# %% [moe] Mixture of experts
class MoE(nn.Module):
    """A router scores each token against every expert; the token goes to its top_k experts, weighted by the softmax of
    those top_k scores (as in gpt-oss). Each expert is a SwiGLU MLP of half the dense hidden size, so a token passes
    through as many weights as the dense MLP's; the layer holds experts / top_k times as many."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.router = nn.Linear(c.d_model, c.experts, bias=False)
        self.experts = nn.ModuleList(MLP(c, swiglu_hidden(c.d_model) // c.top_k) for _ in range(c.experts))
        self.load = None  # fraction of the batch's routed slots each expert got
        self.aux = None

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = self.router(flat)  # (tokens, experts)
        top, chosen = logits.topk(self.c.top_k, dim=-1)
        gates = top.softmax(-1)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            token, slot = (chosen == e).nonzero(as_tuple=True)
            if len(token):
                out.index_add_(0, token, expert(flat[token]) * gates[token, slot, None])
        counts = torch.bincount(chosen.flatten(), minlength=self.c.experts).float()
        self.load = counts / chosen.numel()
        # Switch-style balance: experts · Σ (share of slots) · (mean router probability); 1 when perfectly even
        self.aux = self.c.experts * (self.load * logits.softmax(-1).mean(0)).sum()
        return out.view_as(x)


# %% [model] The model
class Block(nn.Module):
    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.norm1, self.attn, self.norm2 = Norm(c), Attention(c, layer), Norm(c)
        self.mlp = MoE(c) if c.experts else MLP(c)

    def forward(self, x, keep=False):
        x = x + self.attn(self.norm1(x), keep)
        return x + self.mlp(self.norm2(x))


class LadderGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d_model)
        self.wpe = nn.Embedding(c.context, c.d_model) if c.positions == "learned" else None
        self.blocks = nn.ModuleList(Block(c, i) for i in range(c.n_layer))
        self.norm_f = Norm(c)

    def forward(self, idx, keep=False):
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x, keep)
        return self.norm_f(x) @ self.wte.weight.T

    def aux_loss(self):
        return sum(b.mlp.aux for b in self.blocks) / len(self.blocks) if self.c.experts else 0.0


def build(c: Config, seed: int):
    """GPT-2's initialisation (N(0, 0.02), the projections into the residual stream 0.02/√(2·n_layer), gains 1, biases
    and sinks 0). Each weight is drawn from its own generator, seeded by the run's seed and the weight's name, so a
    weight two rungs share starts identical in both."""
    model = LadderGPT(c)
    for name, p in model.named_parameters():
        generator = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
        with torch.no_grad():
            if p.ndim < 2:
                p.copy_(torch.ones_like(p) if name.endswith("gain") else torch.zeros_like(p))
            else:
                std = 0.02 / math.sqrt(2 * c.n_layer) if name.endswith(("attn.o.weight", "proj.weight")) else 0.02
                p.copy_(torch.randn(p.shape, generator=generator) * std)
    return model


# %% [train-step] One training step
def train_step(model, x, y, optimizer):
    """Next-token cross-entropy, plus the experts' balance loss when there are experts; the whole gradient is scaled
    down to norm \`clip\` when it is larger. Returns the loss and the gradient's norm before clipping."""
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    total = loss + model.c.aux_weight * model.aux_loss() if model.c.experts else loss
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.c.clip)
    optimizer.step()
    return loss.detach(), norm
`;export{e as default};
