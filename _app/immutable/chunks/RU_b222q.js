const e=`# AWQ (activation-aware weight quantization), the method in the Efficiency Explainer's AWQ tab, as plain PyTorch.
# Qwen's own release Qwen/Qwen2.5-0.5B-Instruct-AWQ was made with AutoAWQ: before rounding to 4 bits, each linear
# layer's input channels are scaled by s, chosen from the calibration activations, and 1/s is folded into the layer
# that feeds it (an RMSNorm or the previous linear), so the model computes the same function with weights that lose
# less precision where the activations are large. AWQ's kernels need CUDA; on a Mac the integers are unpacked and
# dequantized once. Each "# %% [id]" section is one block of the page.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # the quantization_config of Qwen2.5-0.5B-Instruct-AWQ, and AutoAWQ's search defaults
    bits: int = 4
    group_size: int = 128  # input channels that share one scale and zero
    zero_point: bool = True  # asymmetric: each group has its own zero, so the 16 levels span [min, max]
    version: str = "gemm"  # the packed layout
    n_grid: int = 20  # α = 0, 0.05, …, 0.95
    duo_scaling: bool = True  # s = x̄^α / w̄^(1−α)


ORDER = [0, 2, 4, 6, 1, 3, 5, 7]  # output column of each 4-bit slot of a GEMM word, lowest bits first


# %% [unpack] Unpack the int32 words
def unpack(qweight, qzeros, bits=4):
    """qweight (in, out·bits/32) and qzeros (groups, out·bits/32) hold 32/bits codes per int32 along the output
    dimension, interleaved: slot k of a word is output column ORDER[k] of its 8. Returns codes (in, out) and
    zeros (groups, out) as integers in 0 … 2^bits − 1, in plain column order."""
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    mask = 2**bits - 1
    codes = (qweight[:, :, None] >> shifts) & mask  # (in, out/8, 8 slots)
    zeros = (qzeros[:, :, None] >> shifts) & mask
    slot = torch.tensor(ORDER).argsort()  # slot that holds column j of the eight: 0, 4, 1, 5, 2, 6, 3, 7
    return codes[:, :, slot].reshape(qweight.shape[0], -1), zeros[:, :, slot].reshape(qzeros.shape[0], -1)


# %% [dequantize] Dequantize
def dequantize(codes, zeros, scales, group_size=128):
    """w[o, i] = (code[i, o] − zero[g, o]) · scale[g, o] with g = i // group_size: the weights of the *scaled* layer,
    W·diag(s), as stored. codes (in, out); zeros and scales (groups, out). Returns (out, in) in scales' dtype."""
    zeros = zeros.repeat_interleave(group_size, dim=0)
    scales = scales.repeat_interleave(group_size, dim=0)
    return ((codes - zeros).to(scales.dtype) * scales).T


# %% [quantize] 4-bit with a zero point
def quantize(w, c=Config()):
    """Round each group of group_size input channels of w (out, in) to 16 levels spanning its [min, max]:
    scale = (max − min) / 15, zero = −round(min / scale). Returns the dequantized weights, scales and zeros (out, groups)."""
    shape = w.shape
    w = w.reshape(-1, c.group_size)
    maxq = 2**c.bits - 1
    hi, lo = w.amax(dim=1, keepdim=True), w.amin(dim=1, keepdim=True)
    scales = (hi - lo).clamp(min=1e-5) / maxq
    zeros = (-torch.round(lo / scales)).clamp(0, maxq)
    w = (torch.clamp(torch.round(w / scales) + zeros, 0, maxq) - zeros) * scales
    return w.reshape(shape), scales.view(shape[0], -1), zeros.view(shape[0], -1)


# %% [salience] Which input channels matter
def salience(W, x, group_size=128):
    """x̄: the mean |activation| of each input channel over the calibration tokens, x (tokens, in). Channels with
    large x̄ multiply their weights' rounding errors by large inputs: they are the salient ones.
    w̄: each weight's |w| relative to the largest |w| of its group, averaged over the output rows."""
    x_mean = x.reshape(-1, x.shape[-1]).abs().mean(0)
    w = W.abs().reshape(-1, group_size)
    w_mean = (w / (w.amax(dim=1, keepdim=True) + 1e-6)).view(W.shape).mean(0)
    return x_mean, w_mean


# %% [scales] Scales for one α
def scales_for(x_mean, w_mean, alpha, duo_scaling=True):
    """s = x̄^α / w̄^(1−α), normalized so that max(s) · min(s) = 1. α = 0 ignores the activations; larger α
    scales the salient channels up more before rounding (and their inputs down by the same factor)."""
    if duo_scaling:
        s = (x_mean.pow(alpha) / (w_mean.pow(1 - alpha) + 1e-4)).clamp(min=1e-4)
    else:
        s = x_mean.pow(alpha).clamp(min=1e-4)
    return s / (s.max() * s.min()).sqrt()


# %% [search] Search α on the layer's output
def search(W, x, bias=None, c=Config()):
    """Try every α on the grid: quantize W·diag(s), undo the scaling, and measure the layer's output error on the
    calibration inputs x (tokens, in). Returns the best scales, the best α and the loss of every α."""
    x_mean, w_mean = salience(W, x, c.group_size)
    reference = F.linear(x, W, bias)
    losses = []
    best = (float("inf"), None, None)
    for k in range(c.n_grid):
        alpha = k / c.n_grid
        s = scales_for(x_mean, w_mean, alpha, c.duo_scaling)
        Q = quantize(W * s.view(1, -1), c)[0] / s.view(1, -1)
        loss = (F.linear(x, Q, bias) - reference).float().pow(2).sum().item() / reference.numel()
        losses.append(loss)
        if loss < best[0]:
            best = (loss, alpha, s)
    return best[2], best[1], losses


# %% [fold] Fold 1/s into the layer before
@torch.no_grad()
def fold_into_norm(norm_weight, linears, s):
    """For layers fed by an RMSNorm (q, k and v; gate and up): the norm's weight is divided by s and each layer's
    input columns are multiplied by s, so norm → linear computes exactly what it did before."""
    norm_weight.div_(s)
    for linear in linears:
        linear.weight.mul_(s.view(1, -1))


@torch.no_grad()
def fold_into_linear(previous, linear, s):
    """For down_proj, fed by up_proj (through the SiLU gate's product): up_proj's output rows are divided by s and
    down_proj's input columns multiplied by s."""
    previous.weight[-s.numel() :].div_(s.view(-1, 1))
    if previous.bias is not None:
        previous.bias.div_(s.view(-1))
    linear.weight.mul_(s.view(1, -1))


# %% [model] The whole checkpoint, dequantized
def dequantized_state_dict(state, group_size=128):
    """An AWQ checkpoint's tensors → an ordinary Qwen2 state dict: every <name>.qweight / .qzeros / .scales becomes
    <name>.weight in float32. The norms already carry the folded 1/s, so nothing else changes."""
    out = {}
    for name, tensor in state.items():
        if name.endswith(".qweight"):
            stem = name[: -len(".qweight")]
            codes, zeros = unpack(tensor, state[stem + ".qzeros"])
            out[stem + ".weight"] = dequantize(codes, zeros, state[stem + ".scales"], group_size).float()
        elif not name.endswith((".qzeros", ".scales")):
            out[name] = tensor.float()
    return out


class DequantizedLinear(nn.Module):
    """What the Mac runs in place of AWQ's GEMM kernel: the dequantized weights in an ordinary matrix product."""

    def __init__(self, qweight, qzeros, scales, bias=None, group_size=128):
        super().__init__()
        codes, zeros = unpack(qweight, qzeros)
        self.weight = nn.Parameter(dequantize(codes, zeros, scales, group_size).float(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.float(), requires_grad=False)

    def forward(self, x):  # x: (..., in)
        return F.linear(x, self.weight, self.bias)
`;export{e as default};
