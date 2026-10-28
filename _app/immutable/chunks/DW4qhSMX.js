const e=`# The Hyena operator as in HyenaDNA-tiny (LongSafari/hyenadna-tiny-1k-seqlen-hf), the model in the
# Attention Explainer's Hyena tab. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 128
    d_inner: int = 512  # MLP width
    max_len: int = 1026  # lags the implicit filter is defined at
    emb_dim: int = 5  # positional features per lag: t, and cos/sin at 2 frequencies
    filter_width: int = 64  # hidden width of the filter network
    filter_layers: int = 2  # hidden layers after the first
    sin_freq: float = 10.0  # initial frequency of the filter network's Sin (learned)
    short_kernel: int = 3


# %% [projection] Projection and short causal convolution
class Projection(nn.Module):
    """One linear map gives each token three d_model vectors; a 3-tap causal conv per channel then
    mixes each with its two predecessors. They become the two gates and the values."""

    def __init__(self, c: Config):
        super().__init__()
        self.d = c.d_model
        self.in_proj = nn.Linear(c.d_model, 3 * c.d_model)
        # padding 2, then keep the first T outputs: position t sees t-2, t-1 and t only
        self.short_conv = nn.Conv1d(
            3 * c.d_model, 3 * c.d_model, c.short_kernel, padding=c.short_kernel - 1, groups=3 * c.d_model
        )

    def forward(self, u):  # u: (batch, seq, d_model), after LayerNorm
        t = u.shape[1]
        z = self.short_conv(self.in_proj(u).transpose(1, 2))[..., :t]  # (batch, 3 d_model, seq)
        gate_out, gate_in, v = z.split(self.d, dim=1)  # the model's x[0], x[1] and v
        return gate_in, gate_out, v


# %% [filter] Implicit long filter
class ImplicitFilter(nn.Module):
    """The filter is not stored tap by tap: a small network maps each lag's positional features to
    one value per channel, then an exponential window (a different decay per channel) shapes it.
    So it exists at every lag up to max_len, with a fixed number of parameters."""

    def __init__(self, c: Config):
        super().__init__()
        t = torch.linspace(0, 1, c.max_len)[:, None]  # lag, scaled to [0, 1]
        bands = (c.emb_dim - 1) // 2
        w = 2 * math.pi * torch.arange(c.max_len, dtype=torch.float32)[:, None] / c.max_len
        f = torch.linspace(1e-4, bands - 1, bands)[None]
        self.register_buffer("z", torch.cat([t, torch.cos(-f * w), torch.sin(-f * w)], -1), persistent=False)
        self.register_buffer("t", t, persistent=False)
        self.freq = nn.Parameter(c.sin_freq * torch.ones(1, c.filter_width))  # one Sin, used after each layer
        widths = [c.emb_dim] + [c.filter_width] * (c.filter_layers + 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))
        self.out = nn.Linear(c.filter_width, c.d_model, bias=False)
        decay = torch.linspace(math.log(1e-2) / 1.5, math.log(1e-2) / 0.3, c.d_model).abs()
        self.register_buffer("decay", decay, persistent=False)  # slow to fast, one per channel

    def forward(self, length):
        h = self.z[:length]
        for layer in self.layers:
            h = torch.sin(self.freq * layer(h))
        window = torch.exp(-self.t[:length] * self.decay) + 0.05
        return (self.out(h) * window).T  # (d_model, length): h[c, lag]


# %% [long-conv] Causal long convolution as a T × T matrix
def toeplitz(h, D):
    """Per channel, H[i, j] = h[i − j] for j ≤ i and 0 above the diagonal, plus the skip D on it."""
    d, t = h.shape
    lag = torch.arange(t)[:, None] - torch.arange(t)[None, :]
    H = h[:, lag.clamp(min=0)] * (lag >= 0)
    return H + torch.diag_embed(D[:, None].expand(d, t))  # (d_model, seq, seq)


def long_conv(u, h, D):  # u: (batch, d_model, seq)
    """The model takes this through an FFT, O(T log T); as a matrix it is O(T²) with the same numbers."""
    return torch.einsum("dij,bdj->bdi", toeplitz(h, D), u)


# %% [operator] Hyena operator: gate, long convolution, gate
class HyenaOperator(nn.Module):
    """y = x₂ ⊙ (h ∗ (x₁ ⊙ v) + D ⊙ x₁ ⊙ v): the filter is fixed, the gates come from the input,
    so diag(x₂)·H·diag(x₁) is a data-dependent T × T mixing matrix, like attention weights."""

    def __init__(self, c: Config):
        super().__init__()
        self.projection = Projection(c)
        self.filter = ImplicitFilter(c)
        self.D = nn.Parameter(torch.zeros(c.d_model))  # the skip term ("bias" in the model)
        self.out_proj = nn.Linear(c.d_model, c.d_model)

    def forward(self, u):  # u: (batch, seq, d_model)
        gate_in, gate_out, v = self.projection(u)
        h = self.filter(u.shape[1])
        y = gate_out * long_conv(gate_in * v, h, self.D)
        return self.out_proj(y.transpose(1, 2))


# %% [model] Hyena block
class HyenaBlock(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d_model)
        self.mixer = HyenaOperator(c)
        self.norm2 = nn.LayerNorm(c.d_model)
        self.fc1 = nn.Linear(c.d_model, c.d_inner)
        self.fc2 = nn.Linear(c.d_inner, c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        x = x + self.mixer(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))
`;export{e as default};
