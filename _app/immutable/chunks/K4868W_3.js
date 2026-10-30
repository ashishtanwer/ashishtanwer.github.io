const e=`# QLoRA (Dettmers et al., 2023) on SmolLM2-135M-Instruct, the Post-training Explainer's QLoRA tab, as plain PyTorch:
# the frozen base is stored in 4-bit NormalFloat (NF4), 64 weights per block sharing one absmax; the absmax constants
# are themselves quantized to 8 bits in blocks of 256 (double quantization); every forward pass dequantizes the
# weight and LoRA trains on top exactly as in the LoRA tab. Written to match bitsandbytes' own kernels code for code.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    blocksize: int = 64  # weights per absmax
    nested_blocksize: int = 256  # absmax constants per second-level absmax
    rank: int = 8
    alpha: int = 16
    targets: tuple = ("q_proj", "v_proj")


# %% [nf4] The 4-bit NormalFloat levels
def normal_float_levels(offset: float = 0.9677083) -> torch.Tensor:
    """16 levels for weights that are roughly normal: 8 positive and 7 negative quantiles of N(0, 1) plus an exact
    zero, scaled into [−1, 1]. Each level is equally likely for normal data, so no code is wasted on empty tails."""
    positive = torch.special.ndtri(torch.linspace(offset, 0.5, 9)[:-1].double())  # Φ⁻¹(p), 8 values
    negative = -torch.special.ndtri(torch.linspace(offset, 0.5, 8)[:-1].double())  # 7 values
    levels = torch.cat([negative, torch.zeros(1, dtype=torch.float64), positive]).sort().values
    return (levels / levels.max()).float()  # (16,), from −1 to 1


# %% [quantize] Blockwise NF4 quantization
def quantize_nf4(W: torch.Tensor, levels: torch.Tensor, c: Config):
    """Flatten W, cut it into blocks of 64, scale each block by its largest |w| into [−1, 1] and keep the index of
    the nearest level: 4 bits per weight, two per byte, plus one float32 absmax per block."""
    blocks = W.float().reshape(-1, c.blocksize)  # (n / 64, 64)
    absmax = blocks.abs().max(dim=1).values  # (n / 64,)
    scaled = (blocks * (1.0 / absmax[:, None])).clamp(-1, 1)
    midpoints = (levels[:-1] + levels[1:]) / 2  # the nearest level is the interval between midpoints
    codes = torch.bucketize(scaled.reshape(-1), midpoints).to(torch.uint8)  # 0 … 15
    packed = (codes[0::2] << 4) | codes[1::2]  # the first weight of a pair in the high nibble
    return packed, absmax


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, shape, levels: torch.Tensor, c: Config):
    """Look every code up in the table and multiply by its block's absmax."""
    codes = torch.stack([packed >> 4, packed & 0xF], dim=1).reshape(-1).long()
    return (levels[codes].reshape(-1, c.blocksize) * absmax[:, None]).reshape(shape)


# %% [double-quant] Double quantization of the absmax constants
def dynamic_levels() -> torch.Tensor:
    """bitsandbytes' signed 8-bit dynamic type: a decimal exponent (10⁻⁶ … 10⁰) and a linear fraction that gets
    more bits as the exponent shrinks; 256 sorted levels in [−1, 1]."""
    data = []
    for i in range(7):  # exponent 10^(i − 6)
        bounds = torch.linspace(0.1, 1, 2**i + 1, dtype=torch.float32)
        means = (bounds[:-1] + bounds[1:]) / 2.0
        data += ((10 ** (i - 6)) * means).tolist() + (-(10 ** (i - 6)) * means).tolist()
    data += [0.0, 1.0]
    data += [0.0] * (256 - len(data))
    return torch.tensor(sorted(data), dtype=torch.float32)


def nearest_level(scaled: torch.Tensor, table: torch.Tensor, grid: int = 65536) -> torch.Tensor:
    """The nearest of 256 levels, as bitsandbytes' CPU kernel finds it: the value is snapped to one of 65,536 evenly
    spaced points in [−1, 1] whose nearest levels are precomputed (within ~1.5e-5 of a midpoint it can go either way)."""
    points = -1.0 + (2.0 * torch.arange(grid, dtype=torch.float32)) / (grid - 1)
    nearest = torch.searchsorted((table[:-1] + table[1:]) / 2, points)  # level index for every grid point
    return nearest[((scaled + 1.0) * (0.5 * (grid - 1)) + 0.5).floor().long()].to(torch.uint8)


def quantize_absmax(absmax: torch.Tensor, c: Config):
    """Subtract the mean, then quantize the absmax constants blockwise (256 per block) to the 8-bit dynamic type:
    one byte per 64 weights instead of four, and one float32 per 64 × 256 weights."""
    offset = absmax.mean()
    x = absmax - offset
    table = dynamic_levels()
    n, size = x.numel(), c.nested_blocksize
    starts = range(0, n, size)  # the last block may be short
    absmax2 = torch.stack([x[s : s + size].abs().max() for s in starts])
    scaled = torch.cat([(x[s : s + size] * (1.0 / absmax2[k])).clamp(-1, 1) for k, s in enumerate(starts)])
    return nearest_level(scaled, table), absmax2, offset


def dequantize_absmax(codes: torch.Tensor, absmax2: torch.Tensor, offset: torch.Tensor, c: Config):
    table = dynamic_levels()
    scale = absmax2.repeat_interleave(c.nested_blocksize)[: codes.numel()]
    return table[codes.long()] * scale + offset


# %% [linear4bit] A 4-bit linear layer
class Linear4bit(nn.Module):
    """Holds only the packed codes and the double-quantized absmax; dequantizes W on every call (float32 here)."""

    def __init__(self, linear: nn.Linear, c: Config):
        super().__init__()
        self.c, self.shape = c, tuple(linear.weight.shape)
        self.register_buffer("levels", normal_float_levels())
        packed, absmax = quantize_nf4(linear.weight.detach(), self.levels, c)
        codes, absmax2, offset = quantize_absmax(absmax, c)
        for name, value in (("packed", packed), ("absmax_codes", codes), ("absmax2", absmax2), ("offset", offset)):
            self.register_buffer(name, value)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def weight(self):
        absmax = dequantize_absmax(self.absmax_codes, self.absmax2, self.offset, self.c)
        return dequantize_nf4(self.packed, absmax, self.shape, self.levels, self.c)

    def forward(self, x):
        return F.linear(x, self.weight().to(x.dtype), self.bias)


# %% [qlora] LoRA on the 4-bit base
class QLoRALinear(nn.Module):
    """h = dequant(W₄) x + (α/r) · B A x: the frozen base is 4-bit, the adapters A (r × in) and B (out × r) are
    float32 and the only weights with gradients, as in the LoRA tab."""

    def __init__(self, base: Linear4bit, c: Config):
        super().__init__()
        out_features, in_features = base.shape
        self.base = base
        self.A = nn.Parameter(torch.empty(c.rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, c.rank))  # zero: the first output is the 4-bit model's
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = c.alpha / c.rank

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T


# %% [merge] Merging into a 4-bit weight
@torch.no_grad()
def merge_4bit(layer: QLoRALinear) -> Linear4bit:
    """W′ = dequant(W₄) + (α/r)·B A, quantized back to NF4. Unlike the LoRA tab's float merge this rounds W′ to the
    16 levels again, so the merged layer no longer computes exactly what the adapted layer did."""
    merged = nn.Linear(layer.base.shape[1], layer.base.shape[0], bias=layer.base.bias is not None)
    merged.weight.copy_(layer.base.weight() + layer.scale * layer.B @ layer.A)
    if layer.base.bias is not None:
        merged.bias.copy_(layer.base.bias)
    return Linear4bit(merged, layer.base.c)
`;export{e as default};
