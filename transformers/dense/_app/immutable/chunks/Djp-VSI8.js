const n=`# ConvNeXt's depthwise convolution as in ConvNeXt-T, the model in the Attention Explainer's
# ConvNeXt tab. Each "# %% [id]" section is one step of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dim: int = 768  # channels in stage 4, which sees a 7 × 7 grid of tokens at 224 × 224
    kernel_size: int = 7
    layer_scale_init: float = 1e-6


# %% [dwconv] Depthwise 7 × 7 convolution
def depthwise(dim, kernel_size=7):
    """groups=dim: every channel has its own k × k kernel and bias, and channels never mix.
    Zero padding keeps the grid size."""
    return nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)


# %% [matrix] The same convolution as one T × T matrix per channel
def conv_matrices(kernel, height, width):
    """kernel: (channels, k, k) → (channels, H·W, H·W) on row-major tokens. Entry (i, j) is the
    tap that reads token j when computing token i, or 0 when j is outside i's k × k window.
    Conv2d is a cross-correlation, so token i at (r, c) reads (r + a - k//2, c + b - k//2) with
    kernel[a, b]."""
    k = kernel.shape[-1]
    rows = torch.arange(height).repeat_interleave(width)  # token i sits at (rows[i], cols[i])
    cols = torch.arange(width).repeat(height)
    a = rows[None, :] - rows[:, None] + k // 2  # kernel row for each (i, j)
    b = cols[None, :] - cols[:, None] + k // 2
    inside = (a >= 0) & (a < k) & (b >= 0) & (b < k)
    return kernel[:, a.clamp(0, k - 1), b.clamp(0, k - 1)] * inside


def mix_with_matrices(x, matrices, bias):
    """Each channel's own matrix mixes that channel's tokens, then its bias is added."""
    b, c, h, w = x.shape
    out = torch.einsum("cij,bcj->bci", matrices, x.flatten(2)) + bias[None, :, None]
    return out.view(b, c, h, w)


# %% [block] ConvNeXt block
class ConvNeXtBlock(nn.Module):
    """The depthwise convolution mixes tokens; LayerNorm and a 4× MLP mix channels per token;
    a learned layer scale shrinks the branch before the residual add."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.dwconv = depthwise(c.dim, c.kernel_size)
        self.norm = nn.LayerNorm(c.dim, eps=1e-6)
        self.fc1 = nn.Linear(c.dim, 4 * c.dim)
        self.fc2 = nn.Linear(4 * c.dim, c.dim)
        self.gamma = nn.Parameter(c.layer_scale_init * torch.ones(c.dim))

    def forward(self, x):  # x: (batch, channels, height, width)
        y = self.dwconv(x).permute(0, 2, 3, 1)  # channels last for the per-token MLP
        y = self.gamma * self.fc2(F.gelu(self.fc1(self.norm(y))))
        return x + y.permute(0, 3, 1, 2)
`;export{n as default};
