const e=`# Cross-attention as in T5-small's decoder, the model in the Attention Explainer's Cross tab.
# Each "# %% [id]" section is one step of the diagram. Queries come from the decoder; keys and
# values come from the encoder's output, so each head's weights are (decoder × encoder tokens).
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64  # T5 sets the head size on its own: 8 heads × 64 = 512
    eps: float = 1e-6


# %% [norm] Decoder input: RMSNorm
class RMSNorm(nn.Module):
    """T5's layer norm: divide by the root mean square, then scale. No mean, no bias."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):  # x: (batch, n, d_model), the decoder's residual stream
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# %% [projections] Q from the decoder, K and V from the encoder
class Projections(nn.Module):
    """The only change from self-attention: keys and values project the encoder's output."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        inner = c.n_heads * c.d_head
        self.q = nn.Linear(c.d_model, inner, bias=False)
        self.k = nn.Linear(c.d_model, inner, bias=False)
        self.v = nn.Linear(c.d_model, inner, bias=False)

    def split(self, z):  # (batch, length, heads · d_head) → (batch, heads, length, d_head)
        return z.view(z.shape[0], z.shape[1], self.n_heads, -1).transpose(1, 2)

    def forward(self, x, memory):  # x: (batch, n, d_model) decoder; memory: (batch, m, d_model)
        return self.split(self.q(x)), self.split(self.k(memory)), self.split(self.v(memory))


# %% [attention] Cross-attention: every decoder token reads every encoder token
def cross_attention(q, k, v, keep=None):
    """No causal mask: the whole source sentence is already known. T5 has no 1/√d either; it
    folds that scale into its weights' initialization. \`keep\` (batch, m) is False on padding."""
    scores = q @ k.transpose(-2, -1)  # (batch, heads, n, m)
    if keep is not None:
        scores = scores.masked_fill(~keep[:, None, None, :], float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights  # each head's output (batch, heads, n, d_head), and its weights


# %% [output] Concatenate heads and project
class Output(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.o = nn.Linear(c.n_heads * c.d_head, c.d_model, bias=False)  # W_O

    def forward(self, heads):  # heads: (batch, heads, n, d_head)
        b, h, n, d = heads.shape
        return self.o(heads.transpose(1, 2).reshape(b, n, h * d))


# %% [model] Cross-attention sublayer of a decoder block
class CrossAttention(nn.Module):
    """x + W_O · CrossAttention(RMSNorm(x), memory): the step between a T5 decoder block's
    self-attention and its MLP. \`memory\` is the encoder's last hidden state, the same for
    every decoder block; each block projects it with its own K and V."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.projections = Projections(c)
        self.output = Output(c)

    def forward(self, x, memory, keep=None):
        q, k, v = self.projections(self.norm(x), memory)
        heads, _ = cross_attention(q, k, v, keep)
        return x + self.output(heads)
`;export{e as default};
