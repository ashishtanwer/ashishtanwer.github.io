const n=`# FNet's Fourier token mixing as in google/fnet-base, the model in the Attention Explainer's FNet tab.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    d_model: int = 768
    d_ff: int = 3072
    eps: float = 1e-12


# %% [dft] DFT matrices: F = C − iS
def dft(n, dtype=torch.float64):
    """The n-point DFT matrix F[j, k] = exp(−2πi jk/n), split into its real and imaginary parts.

    Row k is a wave with k cycles over the n positions: C holds its cosine, S its sine.
    """
    index = torch.arange(n, dtype=dtype)
    angle = 2 * math.pi * torch.outer(index, index) / n
    return angle.cos(), angle.sin()  # F = C − iS


# %% [fourier] Fourier mixing: Re(F_T X F_D)
def fourier_mix(x):
    """FNet's attention replacement: a 2-D DFT over tokens and hidden dims, keeping the real part.

    (C_T − iS_T) X (C_D − iS_D) has real part C_T X C_D − S_T X S_D: the token-mixing matrices
    act on the rows, the hidden-mixing matrices on the columns, and nothing here is learned.
    """
    t, d = x.shape[-2:]
    cos_t, sin_t = (m.to(x.dtype) for m in dft(t))
    cos_d, sin_d = (m.to(x.dtype) for m in dft(d))
    return cos_t @ x @ cos_d - sin_t @ x @ sin_d  # == torch.fft.fftn(x, dim=(-2, -1)).real


# %% [model] FNet layer: Fourier mixing, then the feed-forward network
class FNetLayer(nn.Module):
    """Post-norm, like BERT: Add & LayerNorm after the Fourier mixing and after the FFN."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.mixing_norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.ff_in = nn.Linear(c.d_model, c.d_ff)
        self.ff_out = nn.Linear(c.d_ff, c.d_model)
        self.output_norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, x):  # x: (batch, seq, d_model), the previous layer's output
        mixed = self.mixing_norm(x + fourier_mix(x))
        hidden = nn.functional.gelu(self.ff_in(mixed), approximate="tanh")  # gelu_new
        return self.output_norm(mixed + self.ff_out(hidden))
`;export{n as default};
