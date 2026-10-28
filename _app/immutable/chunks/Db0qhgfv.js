const n=`# The Conformer block as in facebook/wav2vec2-conformer-rope-large-960h-ft, the model in the Attention
# Explainer's Conformer tab. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 1024
    n_heads: int = 16  # 1024 / 16 = 64 dimensions per head
    d_ff: int = 4096
    kernel: int = 31  # depthwise taps: 15 frames each side, 20 ms per frame
    rope_base: float = 10000.0


# %% [rope] Rotary position embeddings
def rope_tables(t, head_dim, base=10000.0):
    """cos and sin of position × frequency, one frequency per pair of dimensions: (t, head_dim)."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(t).float(), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate(x, cos, sin, n_heads):
    """Turn each head's (first half, second half) pairs by the position's angles."""
    b, t, d = x.shape
    x = x.view(b, t, n_heads, d // n_heads)
    half = x.shape[-1] // 2
    turned = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos[:, None] + turned * sin[:, None]).view(b, t, d)


# %% [attention] Self-attention with RoPE
class SelfAttention(nn.Module):
    """transformers' rotary Conformer rotates the block input before the query and key projections;
    values come from the unrotated input. No mask: every frame attends to every frame, with
    weights that depend on the audio."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads, self.base = c.n_heads, c.rope_base
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, frames, d_model), after LayerNorm
        b, t, d = x.shape
        cos, sin = rope_tables(t, d // self.n_heads, self.base)
        rotated = rotate(x, cos, sin, self.n_heads)
        q, k, v = (z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in (self.q(rotated), self.k(rotated), self.v(x)))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (batch, heads, t, t)
        heads = weights @ v
        return self.out(heads.transpose(1, 2).reshape(b, t, d)), weights


# %% [pointwise-glu] Pointwise conv and GLU
class PointwiseGLU(nn.Module):
    """LayerNorm, a 1 × 1 conv to 2 × d_model channels (a Linear on each frame), then GLU: a ⊙ σ(b)."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.LayerNorm(c.d_model)
        self.conv = nn.Conv1d(c.d_model, 2 * c.d_model, kernel_size=1, bias=False)

    def forward(self, x):  # x: (batch, frames, d_model)
        a, b = self.conv(self.norm(x).transpose(1, 2)).chunk(2, dim=1)
        return (a * torch.sigmoid(b)).transpose(1, 2)


# %% [depthwise] Depthwise conv as a banded T × T matrix
def depthwise_matrix(kernel, t):
    """Per channel, M[i, j] = kernel[j − i + pad]: the same taps on every row, shifted one frame per
    row, zero beyond the kernel (and past the sequence's ends, which is the conv's zero padding)."""
    pad = (kernel.shape[-1] - 1) // 2
    offset = torch.arange(t)[None, :] - torch.arange(t)[:, None] + pad  # j − i + pad
    inside = (offset >= 0) & (offset < kernel.shape[-1])
    return kernel[:, offset.clamp(0, kernel.shape[-1] - 1)] * inside  # (channels, t, t)


class Depthwise(nn.Module):
    """One kernel per channel and no bias: the frames of a channel mix only with each other."""

    def __init__(self, c: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(c.d_model, 1, c.kernel))  # Conv1d(groups=d_model) layout

    def forward(self, u):  # u: (batch, frames, d_model), the GLU output
        mixing = depthwise_matrix(self.weight[:, 0], u.shape[1])
        return torch.einsum("cij,bjc->bic", mixing, u)


# %% [norm-swish] BatchNorm and Swish
class NormSwish(nn.Module):
    """At inference BatchNorm is a fixed affine map per channel, from its running statistics."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.BatchNorm1d(c.d_model)

    def forward(self, y):  # y: (batch, frames, d_model)
        return F.silu(self.norm(y.transpose(1, 2))).transpose(1, 2)  # Swish = SiLU: y · σ(y)


# %% [pointwise] Output pointwise conv
class Pointwise(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.conv = nn.Conv1d(c.d_model, c.d_model, kernel_size=1, bias=False)

    def forward(self, z):  # z: (batch, frames, d_model)
        return self.conv(z.transpose(1, 2)).transpose(1, 2)


# %% [model] Conformer block: half FFN, self-attention, convolution module, half FFN
class ConvModule(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.pointwise_glu = PointwiseGLU(c)
        self.depthwise = Depthwise(c)
        self.norm_swish = NormSwish(c)
        self.pointwise = Pointwise(c)

    def forward(self, x):
        return self.pointwise(self.norm_swish(self.depthwise(self.pointwise_glu(x))))


class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_ff)
        self.down = nn.Linear(c.d_ff, c.d_model)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class ConformerBlock(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.ffn1_norm, self.ffn1 = nn.LayerNorm(c.d_model), FeedForward(c)
        self.attn_norm, self.attn = nn.LayerNorm(c.d_model), SelfAttention(c)
        self.conv = ConvModule(c)  # brings its own LayerNorm
        self.ffn2_norm, self.ffn2 = nn.LayerNorm(c.d_model), FeedForward(c)
        self.final_norm = nn.LayerNorm(c.d_model)

    def forward(self, x):  # x: (batch, frames, d_model)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        x = x + self.attn(self.attn_norm(x))[0]
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)
`;export{n as default};
