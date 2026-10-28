const e=`# PoolFormer's pooling token mixer as in PoolFormer-S12, the model in the Attention Explainer's
# PoolFormer tab. Each "# %% [id]" section is one step of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dim: int = 512  # channels in stage 4, which sees a 7 × 7 grid of tokens at 224 × 224
    pool_size: int = 3
    mlp_ratio: float = 4.0
    layer_scale_init: float = 1e-5


# %% [norm] GroupNorm with one group
class Norm(nn.Module):
    """Normalizes each image over all its channels and grid cells at once, then scales and
    shifts each channel."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):  # x: (batch, channels, height, width)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


# %% [pool] Pooling token mixer
def pool_mixer(x, pool_size=3):
    """Each token becomes the mean of its pool_size × pool_size neighbourhood, minus itself.
    Cells outside the grid are left out of the mean, not counted as zeros."""
    pooled = F.avg_pool2d(x, pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
    return pooled - x


# %% [matrix] The same mixer as one T × T matrix
def pool_matrix(height, width, pool_size=3):
    """Row i of the (H·W) × (H·W) matrix: 1/|N(i)| on each neighbour of token i (itself
    included), minus 1 on i. Tokens are grid cells in row-major order. Every row sums to zero."""
    rows = torch.arange(height).repeat_interleave(width)  # token i sits at (rows[i], cols[i])
    cols = torch.arange(width).repeat(height)
    reach = pool_size // 2
    near = ((rows[:, None] - rows[None, :]).abs() <= reach) & ((cols[:, None] - cols[None, :]).abs() <= reach)
    near = near.float()
    return near / near.sum(dim=1, keepdim=True) - torch.eye(height * width)


def mix_with_matrix(x, matrix):
    """The same matrix mixes every channel: no queries, no keys, no softmax."""
    b, c, h, w = x.shape
    return (x.flatten(2) @ matrix.T).view(b, c, h, w)


# %% [block] PoolFormer block
class PoolFormerBlock(nn.Module):
    """A MetaFormer block: the pooling mixer where attention would be, then a per-token MLP made
    of 1 × 1 convolutions. Each branch is scaled per channel by a learned layer scale."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        hidden = int(c.dim * c.mlp_ratio)
        self.norm1 = Norm(c.dim)
        self.norm2 = Norm(c.dim)
        self.fc1 = nn.Conv2d(c.dim, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, c.dim, 1)
        self.gamma1 = nn.Parameter(c.layer_scale_init * torch.ones(c.dim))
        self.gamma2 = nn.Parameter(c.layer_scale_init * torch.ones(c.dim))
        self.pool_size = c.pool_size

    def forward(self, x):  # x: (batch, channels, height, width)
        x = x + self.gamma1[:, None, None] * pool_mixer(self.norm1(x), self.pool_size)
        return x + self.gamma2[:, None, None] * self.fc2(F.gelu(self.fc1(self.norm2(x))))
`;export{e as default};
