const e=`# FP8, the method in the Efficiency Explainer's FP8 tab, written as plain PyTorch blocks.
# An FP8 number has 1 sign bit, then exponent and mantissa bits: E4M3 (4 + 3, largest 448) for weights and
# activations, E5M2 (5 + 2, largest 57344) where range matters more than precision. A tensor is divided by a scale
# (max |x| / 448 for E4M3) so its largest value lands on the format's largest, cast, and multiplied back after the
# product. Being floating point, the 8 bits are spread over ~18 binades with 8 values each, so small values keep their
# relative precision where int8's 255 evenly spaced levels lose it. Shown on Qwen2.5-0.5B-Instruct; on a Mac the casts
# use torch.float8_e4m3fn and the products run on the dequantized values, which is what torch._scaled_mm computes on
# the CPU and what an FP8 GEMM computes up to the order of its sums. Each "# %% [id]" section is one block of the page.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dtype: torch.dtype = torch.float8_e4m3fn  # torch.float8_e5m2 for the wider, coarser format
    scheme: str = "tensor"  # "tensor", "channel" (rows and tokens), "block" (128 × 128 weights, 1 × 128 inputs)
    block: int = 128
    activations: str = "dynamic"  # a scale per forward pass, or "static" (calibrated once), or "none" (weights only)


# %% [formats] E4M3 and E5M2
def decode(code: int, exponent_bits: int, mantissa_bits: int, finite_only: bool) -> float:
    """The value of one 8-bit code: sign · 2^(e − bias) · (1 + m / 2^M), or 2^(1 − bias) · m / 2^M when the exponent
    field e is 0 (subnormals, down to zero). E5M2 is IEEE-like: an all-ones exponent is ±inf or NaN. E4M3 "fn"
    (finite, NaN only) gives that exponent back to numbers, keeping only S.1111.111 as NaN, so it reaches 448."""
    bias = 2 ** (exponent_bits - 1) - 1
    sign = -1.0 if code >> 7 else 1.0
    e = (code >> mantissa_bits) & (2**exponent_bits - 1)
    m = code & (2**mantissa_bits - 1)
    top = 2**exponent_bits - 1
    if e == top and (not finite_only or m == 2**mantissa_bits - 1):
        return float("nan") if (m or finite_only) else sign * float("inf")
    if e == 0:
        return sign * 2.0 ** (1 - bias) * m / 2**mantissa_bits
    return sign * 2.0 ** (e - bias) * (1 + m / 2**mantissa_bits)


def grid(dtype=torch.float8_e4m3fn):
    """All 256 codes of a format, in code order."""
    exponent_bits, mantissa_bits, finite_only = (4, 3, True) if dtype == torch.float8_e4m3fn else (5, 2, False)
    return torch.tensor([decode(code, exponent_bits, mantissa_bits, finite_only) for code in range(256)])


# %% [cast] Scale and cast, one scale per tensor
def quantize_tensor(x, dtype=torch.float8_e4m3fn):
    """Δ = max |x| / max_fp8 (448 for E4M3), x_fp8 = cast(x · (1/Δ)), saturating at ±max; x ≈ Δ · x_fp8. The cast
    rounds to the nearest FP8 value (ties to even). Multiplying by 1/Δ, as vLLM's per-tensor kernel does."""
    top = torch.finfo(dtype).max
    scale = x.abs().max().float() / top
    return (x.float() * (1.0 / scale)).clamp(-top, top).to(dtype), scale.view(1)


def dequantize(q, scale):
    return q.float() * scale


# %% [scales] A scale per row or per block
def quantize_rows(x, dtype=torch.float8_e4m3fn):
    """One scale per row: per token for activations (rows of x), per output channel for weights (rows of W).
    Scales are kept above 1 / (max_fp8 · 512), as vLLM's per-token kernel does, so an all-zero row stays finite."""
    top = torch.finfo(dtype).max
    scale = (x.abs().amax(dim=-1).float() / top)[:, None].clamp(min=1.0 / (top * 512.0))
    return (x.float() / scale).clamp(-top, top).to(dtype), scale


def quantize_groups(x, group=128, dtype=torch.float8_e4m3fn):
    """Activations in 1 × 128 tiles: one scale per token and per 128 input channels (DeepSeek-V3's recipe)."""
    top = torch.finfo(dtype).max
    tiles = x.reshape(-1, group)
    scale = tiles.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10).float() / top
    q = (tiles / scale).clamp(-top, top).to(dtype)
    return q.reshape(x.shape), scale.reshape(*x.shape[:-1], x.shape[-1] // group)


def quantize_blocks(w, block=128, dtype=torch.float8_e4m3fn):
    """Weights in 128 × 128 blocks, one scale each, stored as its inverse as FP8 checkpoints do (weight_scale_inv)."""
    top = torch.finfo(dtype).max
    rows, cols = w.shape
    tiles = w.reshape(rows // block, block, cols // block, block).permute(0, 2, 1, 3)  # (row blocks, col blocks, 128, 128)
    scale = top / tiles.abs().amax(dim=(-1, -2))
    q = (tiles * scale[..., None, None]).clamp(-top, top).to(dtype)
    return q.permute(0, 2, 1, 3).reshape(rows, cols), scale.reciprocal()


# %% [gemm] The FP8 matrix product
def fp8_matmul(xq, x_scale, wq, w_scale):
    """y = (Δx · x_fp8) (Δw · W_fp8)ᵀ with one scale per tensor, or per row of x and of W. torch._scaled_mm gives these
    exact numbers on the CPU; an FP8 tensor core multiplies the 8-bit codes, sums the products in its own order and
    precision, then applies Δx · Δw, so it differs only in the last bits of each sum."""
    return (xq.float() * x_scale) @ (wq.float() * w_scale.reshape(-1, 1)).T


def fp8_block_matmul(xq, x_scale, wq, w_scale_inv, block=128):
    """Block scales: for each 128-wide slice k of the inputs, y += (x_fp8[:, k] W_fp8[:, k]ᵀ) · Δx[:, k] · Δw[block, k];
    the partial sums are rescaled before they are added, as DeepSeek-V3's kernels promote them to float32."""
    y = torch.zeros(xq.shape[0], wq.shape[0], device=xq.device)
    for k in range(x_scale.shape[-1]):
        cols = slice(k * block, (k + 1) * block)
        partial = xq[:, cols].float() @ wq[:, cols].float().T  # (tokens, out)
        y += partial * x_scale[:, k : k + 1] * w_scale_inv[:, k].repeat_interleave(block)[None, :]
    return y


# %% [linear] An FP8 linear layer
class Fp8Linear(nn.Module):
    """A linear layer whose weights are stored in FP8 with their scales, and whose input is cast to FP8 at every call
    (one dynamic scale per tensor, per token or per 128-channel tile, or a static scale calibrated beforehand)."""

    def __init__(self, linear: nn.Linear, c: Config, input_scale=None):
        super().__init__()
        self.c, self.bias = c, linear.bias
        w = linear.weight.detach().float()
        if c.scheme == "tensor":
            self.wq, self.w_scale = quantize_tensor(w, c.dtype)
        elif c.scheme == "channel":
            self.wq, self.w_scale = quantize_rows(w, c.dtype)
        else:
            self.wq, self.w_scale = quantize_blocks(w, c.block, c.dtype)
        self.input_scale = input_scale  # static: max |x| over the calibration text / max_fp8

    def weight(self):
        """The dequantized weights, (out, in)."""
        if self.c.scheme == "block":
            return self.wq.float() * self.w_scale.repeat_interleave(self.c.block, 0).repeat_interleave(self.c.block, 1)
        return self.wq.float() * self.w_scale.reshape(-1, 1)

    def forward(self, x):
        shape, x = x.shape, x.reshape(-1, x.shape[-1]).float()
        c, top = self.c, torch.finfo(self.c.dtype).max
        if c.activations == "none":  # weight-only FP8: the input stays in 16 or 32 bits
            y = x @ self.weight().T
        elif c.scheme == "block":
            y = fp8_block_matmul(*quantize_groups(x, c.block, c.dtype), self.wq, self.w_scale, c.block)
        else:
            if c.activations == "static":
                xq, x_scale = (x * (1.0 / self.input_scale)).clamp(-top, top).to(c.dtype), self.input_scale
            elif c.scheme == "tensor":
                xq, x_scale = quantize_tensor(x, c.dtype)
            else:
                xq, x_scale = quantize_rows(x, c.dtype)
            y = fp8_matmul(xq, x_scale, self.wq, self.w_scale)
        y = y if self.bias is None else y + self.bias
        return y.reshape(*shape[:-1], -1)


# %% [calibrate] Static activation scales
@torch.no_grad()
def calibrate(model, batches, dtype=torch.float8_e4m3fn):
    """For a static scheme, each linear layer's input scale is fixed once: max |x| over calibration text / max_fp8."""
    seen, hooks = {}, []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".layers." in f".{name}":
            def keep(module, args, name=name):
                seen[name] = torch.maximum(seen.get(name, torch.tensor(0.0)), args[0].abs().max().float())
            hooks.append(module.register_forward_pre_hook(keep))
    for ids in batches:
        model(ids[None])
    for hook in hooks:
        hook.remove()
    return {name: (amax / torch.finfo(dtype).max).view(1) for name, amax in seen.items()}


# %% [model] FP8 on the whole model
@torch.no_grad()
def fp8_model(model, c: Config = Config(), input_scales=None):
    """Every block's seven linear layers (q, k, v, o, gate, up, down) become FP8; embeddings, norms and the output
    head stay in 16 bits, as FP8 checkpoints keep them."""
    for i, layer in enumerate(model.model.layers):
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")), (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                prefix = "self_attn" if parent is layer.self_attn else "mlp"
                scale = None if input_scales is None else input_scales[f"model.layers.{i}.{prefix}.{name}"]
                setattr(parent, name, Fp8Linear(getattr(parent, name), c, scale))
    return model
`;export{e as default};
