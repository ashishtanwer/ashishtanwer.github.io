const e=`# GPTQ, the method in the Efficiency Explainer's GPTQ tab, written as plain PyTorch blocks.
# Qwen's own release Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4 stores every linear layer of the 24 blocks as
# 4-bit integers in groups of 128 input channels, with one fp16 scale and one 4-bit zero per group and
# output channel. GPTQ's kernels need CUDA, so on a Mac the integers are unpacked and dequantized once
# and the model runs on the dequantized weights. Each "# %% [id]" section is one block of the page.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # the quantization_config of Qwen2.5-0.5B-Instruct-GPTQ-Int4
    bits: int = 4
    group_size: int = 128  # input channels that share one scale and zero
    sym: bool = True  # symmetric: the zero is fixed at 8, so the levels are −8 … 7 times the scale
    desc_act: bool = False  # columns in their own order, so g_idx[i] = i // 128
    damp: float = 0.01  # damp_percent: added to the Hessian's diagonal, times its mean
    block_size: int = 128  # columns per lazy batch update


# %% [unpack] Unpack the int32 words
def unpack(qweight, qzeros, bits=4):
    """qweight (in·bits/32, out) and qzeros (groups, out·bits/32), both int32, hold 32/bits codes per word,
    lowest bits first: qweight along the input dimension, qzeros along the output dimension.
    Returns codes (in, out) and zeros (groups, out) as integers in 0 … 2^bits − 1."""
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)
    mask = 2**bits - 1
    codes = (qweight[:, None, :] >> shifts[None, :, None]) & mask  # (in/8, 8, out)
    zeros = (qzeros[:, :, None] >> shifts[None, None, :]) & mask  # (groups, out/8, 8)
    # The "gptq" (v1) checkpoint format stores zero − 1, so a symmetric zero of 8 is written as 7.
    zeros = (zeros + 1) & mask
    return codes.reshape(-1, qweight.shape[1]), zeros.reshape(qzeros.shape[0], -1)


# %% [dequantize] Dequantize
def dequantize(codes, zeros, scales, g_idx):
    """w[o, i] = scale[g, o] · (code[i, o] − zero[g, o]) with g = g_idx[i].
    codes (in, out), zeros and scales (groups, out); returns the (out, in) weight of nn.Linear, in scales' dtype."""
    g = g_idx.long()
    return (scales[g] * (codes - zeros[g]).to(scales.dtype)).T


# %% [quantizer] Scale and zero of one group
def find_params(w, bits=4):
    """One scale and zero per row of w (rows, group_size), symmetric as in AutoGPTQ's Quantizer:
    scale = (max − min) / 15 over a range made symmetric around 0, zero = 8."""
    maxq = 2**bits - 1
    xmin = torch.clamp(w.min(1).values, max=0)
    xmax = torch.clamp(w.max(1).values, min=0)
    xmax = torch.maximum(xmin.abs(), xmax)
    xmin = torch.where(xmin < 0, -xmax, xmin)
    flat = (xmin == 0) & (xmax == 0)
    xmin, xmax = torch.where(flat, -1.0, xmin), torch.where(flat, 1.0, xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.full_like(scale, (maxq + 1) / 2)
    return scale[:, None], zero[:, None]  # (rows, 1) each


def quantize(w, scale, zero, bits=4):
    """Round to the nearest level of the group and map back: scale · (clamp(round(w / scale) + zero) − zero)."""
    q = torch.clamp(torch.round(w / scale) + zero, 0, 2**bits - 1)
    return scale * (q - zero)


# %% [rtn] Round to nearest
def rtn(W, c=Config()):
    """The baseline GPTQ is measured against: every weight rounded on its own to its group's nearest level.
    W (out, in) → the dequantized weights, in the same 4-bit format as GPTQ's."""
    Q = torch.empty_like(W)
    for g0 in range(0, W.shape[1], c.group_size):
        group = W[:, g0 : g0 + c.group_size].float()
        scale, zero = find_params(group, c.bits)
        Q[:, g0 : g0 + c.group_size] = quantize(group, scale, zero, c.bits)
    return Q


# %% [hessian] The layer's Hessian
def hessian(inputs):
    """The layer's output error ‖W X − Ŵ X‖² has Hessian H = 2 X Xᵀ in the weights of each output row.
    inputs: (sequences, tokens, in), the calibration activations that reach this layer; summed over every token
    and divided by the number of sequences, as AutoGPTQ does. Returns (in, in)."""
    x = inputs.reshape(-1, inputs.shape[-1]).float()
    return 2 / inputs.shape[0] * x.T @ x


# %% [gptq] GPTQ: one column at a time, with error compensation
def gptq(W, H, c=Config()):
    """Quantize W (out, in) column by column. After rounding column i, its error (w − q) / [H⁻¹]ᵢᵢ is pushed onto
    the columns not yet rounded through row i of the upper Cholesky factor of H⁻¹, so that they absorb it and
    the layer's output on the calibration inputs changes as little as possible. Each group's scale and zero is
    found when the loop reaches the group, from weights that already carry the earlier corrections.
    Returns the dequantized weights Q (out, in), scales and zeros (out, groups)."""
    W, H = W.clone().float(), H.clone()
    dead = H.diagonal() == 0  # channels no calibration token used
    H[dead, dead] = 1
    W[:, dead] = 0
    H.diagonal().add_(c.damp * H.diagonal().mean())
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    scales, zeros = [], []
    for b0 in range(0, W.shape[1], c.block_size):
        b1 = min(b0 + c.block_size, W.shape[1])
        W1 = W[:, b0:b1].clone()
        errors = torch.zeros_like(W1)
        for j in range(b1 - b0):
            i = b0 + j
            if i % c.group_size == 0:
                scale, zero = find_params(W[:, i : i + c.group_size], c.bits)
                scales.append(scale)
                zeros.append(zero)
            w, d = W1[:, j], Hinv[i, i]
            q = quantize(w[:, None], scale, zero, c.bits).flatten()
            Q[:, i] = q
            error = (w - q) / d
            W1[:, j:] -= error[:, None] @ Hinv[i, i:b1][None, :]  # the rest of this block, right away
            errors[:, j] = error
        W[:, b1:] -= errors @ Hinv[b0:b1, b1:]  # the columns after the block, once per block
    return Q, torch.cat(scales, 1), torch.cat(zeros, 1)


# %% [linear] The quantized linear layer on a Mac
class DequantizedLinear(nn.Module):
    """What the Mac runs instead of a CUDA kernel: the packed integers are dequantized once at load time and the
    layer is an ordinary matrix product. It saves no memory or time; it reproduces the quantized model's outputs."""

    def __init__(self, qweight, qzeros, scales, g_idx, bias=None, bits=4):
        super().__init__()
        codes, zeros = unpack(qweight, qzeros, bits)
        self.weight = nn.Parameter(dequantize(codes, zeros, scales, g_idx).float(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.float(), requires_grad=False)

    def forward(self, x):  # x: (..., in)
        return F.linear(x, self.weight, self.bias)  # (..., out)


# %% [model] The whole checkpoint, dequantized
def dequantized_state_dict(state):
    """A GPTQ checkpoint's tensors → an ordinary Qwen2 state dict: every <name>.qweight / .qzeros / .scales / .g_idx
    becomes <name>.weight in float32; embeddings, norms and biases are copied (the embedding is not quantized)."""
    out = {}
    for name, tensor in state.items():
        if name.endswith(".qweight"):
            stem = name[: -len(".qweight")]
            codes, zeros = unpack(tensor, state[stem + ".qzeros"])
            out[stem + ".weight"] = dequantize(codes, zeros, state[stem + ".scales"], state[stem + ".g_idx"]).float()
        elif not name.endswith((".qzeros", ".scales", ".g_idx")):
            out[name] = tensor.float()
    return out
`;export{e as default};
