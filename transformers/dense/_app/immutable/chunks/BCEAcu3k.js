const n=`# SmoothQuant, the method in the Efficiency Explainer's SmoothQuant tab, written as plain PyTorch blocks.
# W8A8 runs a linear layer's product on 8-bit integers: weights and activations both rounded to 8 bits. A few
# activation channels are far larger than the rest, and one 8-bit scale for a whole activation (per tensor, or per
# token) is set by them, so every other channel rounds to a handful of levels. SmoothQuant divides each input channel
# by s_j and multiplies the matching weight column by s_j (folded into the RMSNorm before the layer), so the product
# is unchanged and the difficulty moves into the weights, which tolerate it. Shown on Qwen2.5-0.5B-Instruct; on a Mac
# the integer arithmetic is simulated in PyTorch. Each "# %% [id]" section is one block of the page.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    bits: int = 8
    alpha: float = 0.5  # migration strength: 0 leaves the activations alone, 1 moves all of their range to the weights
    act_quant: str = "per_tensor"  # one dynamic scale per activation tensor ("per_token": one per token)


# %% [act-scales] Activation maxima per channel
def activation_scales(inputs):
    """max |x_j| of each input channel over every calibration token reaching the layer. inputs: (..., channels)."""
    return inputs.reshape(-1, inputs.shape[-1]).abs().amax(dim=0).float()


# %% [smooth-scales] Smoothing scales
def smoothing_scales(act_scales, weights, alpha=0.5):
    """s_j = max|X_j|^α / max|W_j|^(1−α), with max|W_j| over the column j of every layer reading this input (q, k
    and v share one input, as do gate and up). Dividing X_j by s_j and multiplying W_j by s_j leaves X·Wᵀ unchanged."""
    weight_scales = torch.stack([w.abs().amax(dim=0) for w in weights]).amax(dim=0).clamp(min=1e-5)
    return (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)


# %% [smooth] Fold the scales into the norm and the weights
@torch.no_grad()
def smooth(norm_weight, linears, s):
    """The RMSNorm before the layers is divided by s, so its output is X·diag(s)⁻¹ at no cost; each layer's weight
    columns are multiplied by s, diag(s)·W in the paper's notation. (X·diag(s)⁻¹)·(W·diag(s))ᵀ = X·Wᵀ."""
    norm_weight.div_(s)
    for linear in linears:
        linear.weight.mul_(s.view(1, -1))


# %% [quantize] 8-bit absmax rounding
def quantize_weight(w, bits=8):
    """One scale per output channel (row): Δ = max|w_row| / 127, w ≈ Δ · round(w / Δ). Returns the rounded weights."""
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / (2 ** (bits - 1) - 1)
    return (w / scale).round() * scale


def quantize_activation(x, per="per_tensor", bits=8):
    """One dynamic scale for the whole tensor (or per token): Δ = max|x| / 127, x ≈ Δ · round(x / Δ)."""
    top = x.abs().amax() if per == "per_tensor" else x.abs().amax(dim=-1, keepdim=True)
    scale = top.clamp(min=1e-5) / (2 ** (bits - 1) - 1)
    return (x / scale).round() * scale


# %% [int8-matmul] The product in integers
def int8_linear(x, weight, bias=None, per="per_tensor", bits=8):
    """What an integer unit computes: x_q = round(x / Δx) and W_q = round(W / Δw) as int8, their product accumulated
    in int32, then one rescale by Δx · Δw per output. Simulated with int32 tensors in PyTorch."""
    q_max = 2 ** (bits - 1) - 1
    top = x.abs().amax() if per == "per_tensor" else x.abs().amax(dim=-1, keepdim=True)
    dx = top.clamp(min=1e-5) / q_max  # () or (tokens, 1)
    dw = weight.abs().amax(dim=-1).clamp(min=1e-5) / q_max  # (out,)
    x_q = (x / dx).round().to(torch.int32)  # values in −127 … 127
    w_q = (weight / dw[:, None]).round().to(torch.int32)
    accumulated = x_q @ w_q.T  # int32
    y = accumulated.float() * dx * dw
    return y if bias is None else y + bias


# %% [linear] W8A8 linear layer
class W8A8Linear(nn.Module):
    """A linear layer with 8-bit weights (per output channel) and 8-bit inputs (per tensor or per token), written as
    rounding then an ordinary product: the same numbers int8_linear gets with integers."""

    def __init__(self, linear: nn.Linear, act_quant="per_tensor"):
        super().__init__()
        self.weight = nn.Parameter(quantize_weight(linear.weight.detach().clone()), requires_grad=False)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        self.act_quant = act_quant

    def forward(self, x):
        return F.linear(quantize_activation(x, self.act_quant), self.weight, self.bias)


# %% [model] SmoothQuant on the whole model
@torch.no_grad()
def smoothquant(model, act_scales, alpha=0.5, act_quant="per_tensor", smoothing=True):
    """Smooth every block's two RMSNorm → linear pairs (input_layernorm → q, k, v; post_attention_layernorm → gate,
    up) with scales from the calibration maxima act_scales[<name of q_proj or gate_proj>], then make all seven linear
    layers of every block W8A8. o_proj and down_proj are quantized without smoothing, as in SmoothQuant's Llama recipe.
    smoothing=False gives naive W8A8."""
    for i, layer in enumerate(model.model.layers):
        attention, mlp = layer.self_attn, layer.mlp
        if smoothing:
            qkv = [attention.q_proj, attention.k_proj, attention.v_proj]
            s = smoothing_scales(act_scales[f"model.layers.{i}.self_attn.q_proj"], [l.weight for l in qkv], alpha)
            smooth(layer.input_layernorm.weight, qkv, s)
            fcs = [mlp.gate_proj, mlp.up_proj]
            s = smoothing_scales(act_scales[f"model.layers.{i}.mlp.gate_proj"], [l.weight for l in fcs], alpha)
            smooth(layer.post_attention_layernorm.weight, fcs, s)
        for parent, names in ((attention, ("q_proj", "k_proj", "v_proj", "o_proj")), (mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                setattr(parent, name, W8A8Linear(getattr(parent, name), act_quant))
    return model
`;export{n as default};
