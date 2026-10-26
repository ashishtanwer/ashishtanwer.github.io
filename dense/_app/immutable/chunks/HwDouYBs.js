const e=`# RWKV-7 0.1B (g1), the model in the Linear attention tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram. RWKV-7 has no attention matrix and no
# KV cache: every head carries one fixed 64 × 64 state from token to token.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65536
    d_model: int = 768
    n_layers: int = 12
    head_size: int = 64  # 12 heads of 64; each head's state is 64 × 64
    d_mlp: int = 3072  # 4 × d_model
    decay_rank: int = 64  # sizes of the low-rank w, a, v-first and g projections
    a_rank: int = 64
    v_rank: int = 32
    gate_rank: int = 128
    eps: float = 1e-5


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token lookup, then block 1's extra LayerNorm (ln0). There is no position input:
    order comes from the recurrence itself."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.ln0 = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, ids):  # ids: (batch, seq)
        return self.ln0(self.token(ids))  # (batch, seq, d_model)


# %% [state-update] WKV-7 state update
def wkv7(r, w, k, v, kk, a):
    """The generalized delta rule, one token at a time. Each head keeps a (value × key) state S:

        S ← S (diag(w) − κ̂ (κ̂ ⊙ a)ᵀ) + v k̃ᵀ,    o = S r

    w decays each key channel, the κ̂ term erases a fraction a of what S held along κ̂,
    v k̃ᵀ writes this token and r reads the result out.
    Inputs are (batch, seq, heads, head_size); returns o (same shape) and the last S."""
    b, t, h, n = r.shape
    S = r.new_zeros(b, h, n, n)  # (batch, heads, value, key)
    out = []
    for i in range(t):
        transition = torch.diag_embed(w[:, i]) - kk[:, i, :, :, None] * (kk[:, i] * a[:, i])[:, :, None, :]
        S = S @ transition + v[:, i, :, :, None] * k[:, i, :, None, :]
        out.append((S @ r[:, i, :, :, None]).squeeze(-1))
    return torch.stack(out, dim=1), S


# %% [mixing-matrix] Token mixing A
def mixing_matrix(r, w, k, kk, a):
    """The same recurrence as one causal matrix per head, as the diagram draws it:

        A[t, i] = k̃_iᵀ G_{i+1} ⋯ G_t r_t,    G_j = diag(w_j) − κ̂_j (κ̂_j ⊙ a_j)ᵀ

    so o_t = Σ_i A[t, i] v_i. Inputs are (seq, heads, head_size) for one sequence;
    returns (heads, seq, seq). Entries can be negative: the delta rule subtracts."""
    t, h, n = r.shape
    G = torch.diag_embed(w) - kk[..., :, None] * (kk * a)[..., None, :]  # (seq, heads, key, key)
    A = r.new_zeros(h, t, t)
    for row in range(t):
        u = r[row]  # G_{i+1} ⋯ G_t r_t, grown one factor per earlier token
        for i in range(row, -1, -1):
            A[:, row, i] = (k[i] * u).sum(-1)
            u = (G[i] @ u[..., None]).squeeze(-1)
    return A


# %% [time-mix] Time mix
def token_shift(x):
    """Each position's previous token, with zeros before the first: (batch, seq, d) → same."""
    return F.pad(x, (0, 0, 1, -1))


class LoRA(nn.Module):
    """A low-rank projection d → rank → d, with an optional activation in the middle."""

    def __init__(self, d: int, rank: int, activation=None, bias=True):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=bias)
        self.activation = activation or nn.Identity()

    def forward(self, x):
        return self.up(self.activation(self.down(x)))


class TimeMix(nn.Module):
    """RWKV-7's replacement for attention: six token-shifted inputs, the WKV-7 state update,
    a per-head GroupNorm, the r·k bonus, the output gate and Wo."""

    def __init__(self, c: Config, layer: int):
        super().__init__()
        d, n = c.d_model, c.head_size
        self.first = layer == 0
        # How far each of r, w, k, v, a and g leans toward the previous token, per channel.
        self.shift = nn.ParameterDict({name: nn.Parameter(torch.zeros(d)) for name in "rwkvag"})
        self.r = nn.Linear(d, d, bias=False)  # receptance: what each token reads from the state
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.w = LoRA(d, c.decay_rank, nn.Tanh())  # decay, before it is squashed into (0.545, 1)
        self.a = LoRA(d, c.a_rank)  # in-context learning rate of the delta rule
        self.g = LoRA(d, c.gate_rank, nn.Sigmoid(), bias=False)  # output gate
        if not self.first:
            self.v_first_mix = LoRA(d, c.v_rank)  # how much of block 1's value to blend back in
        self.k_k = nn.Parameter(torch.zeros(d))  # κ̂ = normalize(k ⊙ k_k): the direction a erases
        self.k_a = nn.Parameter(torch.zeros(d))  # how much a also rescales the written key
        self.r_k = nn.Parameter(torch.zeros(d // n, n))  # the bonus weights, per head
        self.ln_x = nn.GroupNorm(d // n, d, eps=n * 1e-5)  # one group per head
        self.out = nn.Linear(d, d, bias=False)

    def projections(self, x, v_first):
        """The r, w, k, v, a and g bank. x: (batch, seq, d_model), already LayerNormed."""
        b, t, d = x.shape
        h, n = self.r_k.shape
        lerp = {name: x + (token_shift(x) - x) * mix for name, mix in self.shift.items()}
        r = self.r(lerp["r"])
        w = torch.exp(-math.exp(-0.5) * torch.sigmoid(self.w(lerp["w"])))  # in (0.545, 1)
        k = self.k(lerp["k"])
        v = self.v(lerp["v"])
        a = torch.sigmoid(self.a(lerp["a"]))
        g = self.g(lerp["g"])
        kk = F.normalize((k * self.k_k).view(b, t, h, n), dim=-1).view(b, t, d)
        k = k * (1 + (a - 1) * self.k_a)  # k̃
        if self.first:
            v_first = v  # block 1 hands its values to every later block
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v_first_mix(lerp["v"]))
        return r, w, k, v, kk, a, g, v_first

    def forward(self, x, v_first=None):
        b, t, d = x.shape
        h, n = self.r_k.shape
        r, w, k, v, kk, a, g, v_first = self.projections(x, v_first)
        heads = lambda z: z.view(b, t, h, n)
        o, _ = wkv7(heads(r), heads(w), heads(k), heads(v), heads(kk), heads(a))
        o = self.ln_x(o.reshape(b * t, d)).view(b, t, d)
        bonus = (heads(r) * heads(k) * self.r_k).sum(-1, keepdim=True) * heads(v)  # this token's own r·k̃ v
        return self.out((o + bonus.view(b, t, d)) * g), v_first


# %% [channel-mix] Channel mix
class ChannelMix(nn.Module):
    """RWKV-7's MLP: its own token shift, then d_model → 4 × d_model → d_model with squared ReLU."""

    def __init__(self, c: Config):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(c.d_model))
        self.key = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.value = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        x = x + (token_shift(x) - x) * self.shift
        return self.value(torch.relu(self.key(x)) ** 2)


# %% [block] RWKV-7 block
class Block(nn.Module):
    """Pre-norm residual block: x + TimeMix(LN(x)), then x + ChannelMix(LN(x))."""

    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.time_mix = TimeMix(c, layer)
        self.ln2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.channel_mix = ChannelMix(c)

    def forward(self, x, v_first=None):
        mixed, v_first = self.time_mix(self.ln1(x), v_first)
        x = x + mixed
        return x + self.channel_mix(self.ln2(x)), v_first


# %% [output] Probabilities
class Output(nn.Module):
    """Final LayerNorm, then a projection to one logit per vocabulary token (not tied)."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x):
        return self.head(self.norm(x))  # logits: (batch, seq, vocab)


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
class RWKV7(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c, layer) for layer in range(c.n_layers))
        self.output = Output(c)

    def forward(self, ids):
        x, v_first = self.embedding(ids), None
        for block in self.blocks:
            x, v_first = block(x, v_first)
        return self.output(x)


# probabilities = next_token_probabilities(RWKV7()(ids), temperature=0.8, top_k=5)
`;export{e as default};
