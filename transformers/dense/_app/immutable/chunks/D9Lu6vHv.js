const e=`# Moshi (Défossez et al. 2024), kyutai/moshiko-pytorch-bf16, as plain PyTorch in Kyutai's own parameter layout (the
# names of the released model.safetensors load into it as they are). Moshi listens and speaks at the same time, 12.5
# frames a second. At every step it reads 17 streams, summed as embeddings: its own text token (the inner monologue),
# its own 8 Mimi codes and the user's 8 Mimi codes. A 7B temporal transformer over the steps gives one 4,096-wide state;
# the text token is read from it, then a small depth transformer writes Moshi's 8 codes one after another, each of its
# 8 steps with its own weights. Acoustic codes run one step behind the semantic code (the acoustic delay).
# Each "# %% [id]" section is one part of the Moshi tab's diagram; Mimi, the codec, is the Tokenization tab's.
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    dim: int = 4096
    layers: int = 32
    heads: int = 32
    hidden: int = 11264  # SiLU-gated MLP: linear_in gives 2 × 11,264
    context: int = 3000  # steps a step can attend to: 4 minutes
    max_period: float = 10000.0
    text_card: int = 32000  # SentencePiece pieces; 32000 is the initial token
    card: int = 2048  # Mimi codes per codebook; 2048 is the initial token
    n_q: int = 16  # audio streams read: Moshi's 8, then the user's 8
    dep_q: int = 8  # audio streams written: Moshi's 8
    delays: tuple = (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1)  # text, Moshi's 8 codes, the user's 8
    depth_dim: int = 1024
    depth_layers: int = 6
    depth_heads: int = 16
    depth_hidden: int = 2816
    eps: float = 1e-8
    text_padding: int = 3  # PAD, between words
    end_of_padding: int = 0  # EPAD, the step before a word
    temp_text: float = 0.7
    top_k_text: int = 25
    temp: float = 0.8
    top_k: int = 250


# %% [streams] Seventeen streams in, nine out, with their delays
def initial(c: Config):
    return torch.tensor([c.text_card] + [c.card] * c.n_q)


def place_user(grid, n, codes, c: Config):
    """The user's frame n (8 Mimi codes) goes in at step n + its delay: the semantic code at step n, the acoustic
    ones at n + 1. grid (17, steps + 2) holds, in column p, what step p reads."""
    for j, code in enumerate(codes):
        grid[1 + c.dep_q + j, n + c.delays[1 + c.dep_q + j]] = code


def step_input(grid, n, c: Config):
    """What step n reads: column n, with a stream's initial token while n is not past its delay."""
    column = grid[:, n].clone()
    first = n <= torch.tensor(c.delays)
    column[first] = initial(c)[first]
    return column


def place_moshi(grid, n, text, codes):
    """Step n writes Moshi's text token and 8 codes into column n + 1, which step n + 1 reads."""
    grid[0, n + 1] = text
    grid[1:1 + len(codes), n + 1] = codes


def moshi_frame(grid, n, c: Config):
    """After step n ≥ 1, Moshi's frame n − 1: its text token and semantic code from step n − 1, its acoustic codes
    (delayed by one) from step n."""
    streams = torch.arange(1 + c.dep_q)
    return grid[streams, n + torch.tensor(c.delays[:1 + c.dep_q])]


# %% [embedding] One vector per step: the sum of 17 embeddings
class Embedding(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.text_emb = nn.Embedding(c.text_card + 1, c.dim)
        self.emb = nn.ModuleList(nn.Embedding(c.card + 1, c.dim) for _ in range(c.n_q))

    def forward(self, tokens):
        """tokens (batch, 17, steps) → (batch, steps, dim): Moshi's and the user's 16 codes, then the text token."""
        x = sum(table(tokens[:, 1 + s]) for s, table in enumerate(self.emb))
        return x + self.text_emb(tokens[:, 0])


# %% [rope] Rotary positions: pairs of coordinates turned by the step's angle
def rope(x, offset, max_period):
    """x (batch, heads, steps, head_dim). Coordinates (2i, 2i + 1) are the real and imaginary parts of a complex
    number, turned by (offset + t) · max_period^(−2i / head_dim) radians; computed in float32."""
    d = x.shape[-1]
    freqs = torch.exp(torch.arange(d // 2, dtype=torch.float32) * (-math.log(max_period) * 2 / d))
    angle = (offset + torch.arange(x.shape[-2], dtype=torch.float32))[:, None] * freqs
    pairs = x.float().view(*x.shape[:-1], d // 2, 2)
    re, im = pairs[..., 0], pairs[..., 1]
    turned = torch.stack([re * angle.cos() - im * angle.sin(), re * angle.sin() + im * angle.cos()], -1)
    return turned.view(x.shape).to(x.dtype)


# %% [attention] Self-attention over the steps so far
def per_step(x, weight, steps):
    """x (batch, t, in) through a linear layer; with \`steps\` > 1 the weight stacks one matrix per step and position
    t uses the t-th (the depth transformer's weights per step)."""
    if steps == 1:
        return x @ weight.T
    return torch.einsum("bti,toi->bto", x, weight.view(steps, -1, weight.shape[-1])[: x.shape[1]])


class Attention(nn.Module):
    def __init__(self, dim, heads, context, steps=1, max_period=None):
        super().__init__()
        self.heads, self.context, self.steps, self.max_period = heads, context, steps, max_period
        self.in_proj_weight = nn.Parameter(torch.empty(steps * 3 * dim, dim))
        self.out_proj = nn.Linear(dim, steps * dim, bias=False)

    def forward(self, x, offset=0):
        """x (batch, t, dim) → (batch, t, dim), each step attending to itself and the \`context\` − 1 before it."""
        b, t, d = x.shape
        q, k, v = per_step(x, self.in_proj_weight, self.steps).view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        if self.max_period is not None:
            q, k = rope(q, offset, self.max_period), rope(k, offset, self.max_period)
        delta = torch.arange(t)[:, None] - torch.arange(t)[None]
        allowed = (delta >= 0) & (delta < self.context)
        out = F.scaled_dot_product_attention(q, k, v, allowed)  # softmax(q·kᵀ/√d) over allowed keys
        return per_step(out.transpose(1, 2).reshape(b, t, d), self.out_proj.weight, self.steps)


# %% [mlp] The SiLU-gated MLP
class Gating(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.linear_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, up = self.linear_in(x).view(*x.shape[:-1], 2, -1).unbind(-2)
        return self.linear_out(F.silu(gate) * up)


# %% [block] One layer: RMS norm, attention, RMS norm, gated MLP, each added back
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x):
        y = x.float()
        return (y * (self.alpha.float() * torch.rsqrt(self.eps + y.pow(2).mean(-1, keepdim=True)))).to(x.dtype)


class Layer(nn.Module):
    def __init__(self, dim, heads, hidden, context, eps, steps=1, max_period=None):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.self_attn = Attention(dim, heads, context, steps, max_period)
        self.steps = steps
        self.gating = Gating(dim, hidden) if steps == 1 else nn.ModuleList(Gating(dim, hidden) for _ in range(steps))

    def forward(self, x, offset=0):
        x = x + self.self_attn(self.norm1(x), offset)
        h = self.norm2(x)
        if self.steps == 1:
            return x + self.gating(h)
        return x + torch.stack([self.gating[i](h[:, i]) for i in range(h.shape[1])], 1)


class Transformer(nn.Module):
    def __init__(self, layers, *args, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(Layer(*args, **kwargs) for _ in range(layers))

    def forward(self, x, offset=0):
        for layer in self.layers:
            x = layer(x, offset)
        return x


# %% [output] The inner monologue: the text token's logits
class TextHead(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.out_norm = RMSNorm(c.dim, c.eps)
        self.text_linear = nn.Linear(c.dim, c.text_card, bias=False)

    def forward(self, x):
        """The temporal transformer's output → (normalised state the depth transformer reads, text logits)."""
        state = self.out_norm(x)
        return state, self.text_linear(state)


# %% [depth] The depth transformer: Moshi's 8 codes for the step, one after another
class Depth(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.depformer_in = nn.ModuleList(nn.Linear(c.dim, c.depth_dim, bias=False) for _ in range(c.dep_q))
        self.depformer_text_emb = nn.Embedding(c.text_card + 1, c.depth_dim)
        self.depformer_emb = nn.ModuleList(nn.Embedding(c.card + 1, c.depth_dim) for _ in range(c.dep_q - 1))
        self.depformer = Transformer(c.depth_layers, c.depth_dim, c.depth_heads, c.depth_hidden, c.dep_q, c.eps,
                                     steps=c.dep_q)
        self.linears = nn.ModuleList(nn.Linear(c.depth_dim, c.card, bias=False) for _ in range(c.dep_q))

    def forward(self, state, tokens):
        """state (batch, 4096) of one step; tokens (batch, k ≤ 8): the step's text token, then its first k − 1 codes.
        Step i reads its own projection of the state plus the token before it → logits for code i, (batch, k, 2048)."""
        steps = tokens.shape[1]
        x = torch.stack([self.depformer_in[i](state) + (self.depformer_text_emb(tokens[:, 0]) if i == 0 else
                         self.depformer_emb[i - 1](tokens[:, i])) for i in range(steps)], 1)
        out = self.depformer(x)
        return torch.stack([self.linears[i](out[:, i]) for i in range(steps)], 1)


# %% [sampling] Drawing a token: temperature, top-k, then the uniform
def top_k(logits, k, temperature):
    """logits (vocab,) → the k most likely ids, most likely first, and softmax(logits / temperature) over them."""
    scaled = logits.double() / temperature
    order = torch.argsort(-scaled, stable=True)[:k]
    return order, torch.softmax(scaled[order], 0)


def draw(ids, probs, u):
    """The id whose slice of [0, 1), laid out most likely first, holds the uniform u."""
    edges = torch.cumsum(probs, 0)
    i = int(torch.searchsorted(edges, torch.tensor([u], dtype=edges.dtype), right=True)[0])
    return int(ids[min(i, len(ids) - 1)])


# %% [model] Moshi: the streams read so far → each step's text logits and state
class Moshi(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        for part in (Embedding(c), TextHead(c), Depth(c)):
            for name, module in part.named_children():
                self.add_module(name, module)  # Kyutai's flat names: text_emb, emb, out_norm, depformer_in, …
        self.transformer = Transformer(c.layers, c.dim, c.heads, c.hidden, c.context, c.eps,
                                       max_period=c.max_period)

    def forward(self, tokens):
        """tokens (batch, 17, steps), what each step read → (state (batch, steps, 4096), text logits)."""
        return TextHead.forward(self, self.transformer(Embedding.forward(self, tokens)))

    def codes(self, state, tokens):
        """The depth transformer's logits for one step's state (Depth.forward)."""
        return Depth.forward(self, state, tokens)
`;export{e as default};
