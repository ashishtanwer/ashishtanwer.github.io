const e=`# Linear attention as RWKV-7 0.1B (g1) does it, the model in the Attention Explainer's Linear tab.
# Each "# %% [id]" section is one step of the diagram. There is no softmax and no KV cache: each
# head keeps one 64 × 64 state, and that recurrence equals a causal T × T mixing matrix.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 768
    head_size: int = 64  # 12 heads of 64; each head's state is 64 × 64
    decay_rank: int = 64  # sizes of the low-rank w, a, v-first and g projections
    a_rank: int = 64
    v_rank: int = 32
    gate_rank: int = 128
    first: bool = False  # block 1 has no v-first blend: it hands its values to later blocks


class LoRA(nn.Module):
    """A low-rank projection d → rank → d, with an optional activation in the middle."""

    def __init__(self, d: int, rank: int, activation=None, bias=True):
        super().__init__()
        self.down = nn.Linear(d, rank, bias=False)
        self.up = nn.Linear(rank, d, bias=bias)
        self.activation = activation or nn.Identity()

    def forward(self, x):
        return self.up(self.activation(self.down(x)))


# %% [projections] r · w · k̃ · v · κ̂ · a
class Projections(nn.Module):
    """Every token makes a receptance r (the query), a key k̃, a value v, a decay w, and the
    delta rule's erase direction κ̂ and rate a, from a mix of itself and the previous token."""

    def __init__(self, c: Config):
        super().__init__()
        d, n = c.d_model, c.head_size
        self.first, self.n = c.first, n
        # How far each of r, w, k, v, a and g leans toward the previous token, per channel.
        self.shift = nn.ParameterDict({name: nn.Parameter(torch.zeros(d)) for name in "rwkvag"})
        self.r = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.w = LoRA(d, c.decay_rank, nn.Tanh())
        self.a = LoRA(d, c.a_rank)
        self.g = LoRA(d, c.gate_rank, nn.Sigmoid(), bias=False)  # output gate, used in [output]
        if not self.first:
            self.v_first_mix = LoRA(d, c.v_rank)  # how much of block 1's value to blend back in
        self.k_k = nn.Parameter(torch.zeros(d))  # κ̂ = normalize(k ⊙ k_k)
        self.k_a = nn.Parameter(torch.zeros(d))  # how much a also rescales the written key

    def forward(self, x, v_first=None):  # x: (batch, seq, d_model), after ln1
        b, t, d = x.shape
        previous = F.pad(x, (0, 0, 1, -1))  # each position's previous token, zeros first
        lerp = {name: x + (previous - x) * mix for name, mix in self.shift.items()}
        r = self.r(lerp["r"])
        w = torch.exp(-math.exp(-0.5) * torch.sigmoid(self.w(lerp["w"])))  # decay in (0.545, 1)
        k = self.k(lerp["k"])
        v = self.v(lerp["v"])
        a = torch.sigmoid(self.a(lerp["a"]))  # in-context learning rate in (0, 1)
        g = self.g(lerp["g"])
        kk = F.normalize((k * self.k_k).view(b, t, -1, self.n), dim=-1).view(b, t, d)
        k = k * (1 + (a - 1) * self.k_a)  # k̃, the key written into the state
        if self.first:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v_first_mix(lerp["v"]))
        return r, w, k, v, kk, a, g, v_first


# %% [state] One 64 × 64 state per head
def wkv7(r, w, k, v, kk, a):
    """The generalized delta rule, one token at a time. Each head keeps a (value × key) state S:

        S_t = S_{t-1} (diag(w_t) − κ̂_t (κ̂_t ⊙ a_t)ᵀ) + v_t k̃_tᵀ,    o_t = S_t r_t

    w decays each key channel, the κ̂ term erases a fraction a of what S held along κ̂, v k̃ᵀ
    writes this token, and r reads the result. Inputs are (batch, seq, heads, head_size);
    returns o (same shape) and the last S, (batch, heads, head_size, head_size)."""
    b, t, h, n = r.shape
    S = r.new_zeros(b, h, n, n)
    out = []
    for i in range(t):
        transition = torch.diag_embed(w[:, i]) - kk[:, i, :, :, None] * (kk[:, i] * a[:, i])[:, :, None, :]
        S = S @ transition + v[:, i, :, :, None] * k[:, i, :, None, :]
        out.append((S @ r[:, i, :, :, None]).squeeze(-1))
    return torch.stack(out, dim=1), S


# %% [mixing] The same recurrence as a T × T matrix
def mixing_matrix(r, w, k, kk, a):
    """Unrolling the recurrence gives one causal matrix per head, the one the head card draws:

        A[t, i] = k̃_iᵀ G_{i+1} ⋯ G_t r_t,    G_j = diag(w_j) − κ̂_j (κ̂_j ⊙ a_j)ᵀ

    so o_t = Σ_{i ≤ t} A[t, i] v_i, attention without a softmax. Inputs are (seq, heads,
    head_size) for one sequence; returns (heads, seq, seq). Entries can be negative."""
    t, h, n = r.shape
    G = torch.diag_embed(w) - kk[..., :, None] * (kk * a)[..., None, :]  # (seq, heads, n, n)
    A = r.new_zeros(h, t, t)
    for row in range(t):
        u = r[row]  # G_{i+1} ⋯ G_t r_t, one more factor per earlier token
        for i in range(row, -1, -1):
            A[:, row, i] = (k[i] * u).sum(-1)
            u = (G[i] @ u[..., None]).squeeze(-1)
    return A


# %% [output] GroupNorm, bonus, gate and W_o
class Output(nn.Module):
    """Each head's o is normalized on its own, this token's own r·k̃ v is added back (the bonus),
    the gate g scales every channel, and W_o maps the heads back to the residual stream."""

    def __init__(self, c: Config):
        super().__init__()
        d, n = c.d_model, c.head_size
        self.r_k = nn.Parameter(torch.zeros(d // n, n))  # the bonus weights, per head
        self.ln_x = nn.GroupNorm(d // n, d, eps=n * 1e-5)  # one group per head
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, o, r, k, v, g):  # o, r, k, v: (batch, seq, heads, head_size); g: (b, s, d)
        b, t, h, n = o.shape
        normed = self.ln_x(o.reshape(b * t, h * n)).view(b, t, h, n)
        bonus = (r * k * self.r_k).sum(-1, keepdim=True) * v
        return self.out((normed + bonus).reshape(b, t, h * n) * g)


# %% [model] RWKV-7 time mix
class TimeMix(nn.Module):
    """RWKV-7's replacement for attention, as one layer: projections, the state recurrence,
    then the output step. Returns the layer output and block 1's values (v-first)."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.projections = Projections(c)
        self.output = Output(c)

    def forward(self, x, v_first=None):
        b, t, d = x.shape
        r, w, k, v, kk, a, g, v_first = self.projections(x, v_first)
        heads = lambda z: z.view(b, t, -1, self.projections.n)
        o, _ = wkv7(heads(r), heads(w), heads(k), heads(v), heads(kk), heads(a))
        return self.output(o, heads(r), heads(k), heads(v), g), v_first
`;export{e as default};
