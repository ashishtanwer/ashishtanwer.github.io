const e=`# Mixed-precision training, as the Pretraining Explainer trains its small GPT with it (4 layers, d_model 256, 4 heads,
# context 256, ≈16.1M parameters, AdamW on TinyStories): the weights and AdamW's state stay in float32 (the master
# copy), and only the arithmetic of the forward and backward passes runs in a narrower format: bfloat16 under
# autocast, float16 under autocast with the loss scaled so small gradients survive, or FP8 for the matmuls of every
# block, e4m3 for weights and activations and e5m2 for gradients, each tensor scaled to fill its format's range.
# scripts/pretraining/record_mixed_precision.py trains every run of the tab with these functions.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    init_scale: float = 2.0**16  # float16 loss scaling: torch.amp.GradScaler's defaults …
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 200  # … except this one (2000 by default, as long as the whole run)
    forward_format: torch.dtype = torch.float8_e4m3fn  # FP8 weights and activations: more precision
    backward_format: torch.dtype = torch.float8_e5m2  # FP8 gradients: more range


# %% [formats] Number formats
def format_table():
    """Bits of exponent and mantissa, the largest value, the smallest normal and subnormal value and the relative
    spacing (machine epsilon) of each format the tab compares."""
    rows = {}
    for name, dtype, exponent, mantissa in [("fp32", torch.float32, 8, 23), ("bf16", torch.bfloat16, 8, 7),
                                            ("fp16", torch.float16, 5, 10), ("e4m3", torch.float8_e4m3fn, 4, 3),
                                            ("e5m2", torch.float8_e5m2, 5, 2)]:
        info = torch.finfo(dtype)
        rows[name] = {"exponent": exponent, "mantissa": mantissa, "max": info.max, "minNormal": info.smallest_normal,
                      "minSubnormal": info.smallest_normal * 2.0**-mantissa, "eps": info.eps}
    return rows


# %% [round] Rounding to a format
def round_to(x, dtype):
    """x rounded to the nearest value of the format (ties to even) and read back as float32: the only error mixed
    precision adds. MPS has no float8 type, so FP8 rounds on the CPU; PyTorch's FP8 casts saturate at the format's
    largest finite value (e4m3 has no infinity), e5m2 overflows to inf."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and x.device.type == "mps":
        return x.cpu().to(dtype).to(torch.float32).to(x.device)
    return x.to(dtype).to(torch.float32)


# %% [scale] Per-tensor scaling for FP8
def fp8_scale(x, dtype):
    """FP8 covers few binades (e4m3: 2⁻⁹ … 448), so each tensor is multiplied by a scale that maps its largest
    magnitude onto the format's largest value before the cast, and divided by it after: the rounding is relative to
    the tensor's own range. This is current (just-in-time) scaling, from this very tensor's amax."""
    amax = x.detach().abs().max().float().clamp(min=1e-12)
    return torch.finfo(dtype).max / amax


def quantize(x, dtype):
    s = fp8_scale(x, dtype)
    return round_to(x * s, dtype) / s  # FP8 values, held as float32 for the emulated matmul


# %% [fp8-linear] An FP8 matmul
class FP8Linear(torch.autograd.Function):
    """y = x Wᵀ with both inputs rounded to e4m3 and the product accumulated in float32, as FP8 tensor cores do; in
    the backward pass the incoming gradient is rounded to e5m2 and both products (for x and for W) use the rounded
    copies. W itself, its gradient and AdamW's update stay float32."""

    @staticmethod
    def forward(ctx, x, weight, c: Config):
        x8, w8 = quantize(x, c.forward_format), quantize(weight, c.forward_format)
        ctx.save_for_backward(x8, w8)
        ctx.c = c
        return x8 @ w8.T

    @staticmethod
    def backward(ctx, grad_y):
        x8, w8 = ctx.saved_tensors
        g8 = quantize(grad_y, ctx.c.backward_format)
        grad_x = g8 @ w8  # (…, in)
        grad_w = g8.flatten(0, -2).T @ x8.flatten(0, -2)  # (out, in)
        return grad_x, grad_w, None


def fp8_linear(x, weight, c: Config):
    return FP8Linear.apply(x, weight, c)


# %% [loss-scaling] Dynamic loss scaling
class LossScaler:
    """float16's smallest subnormal is 2⁻²⁴ ≈ 6e-8, and many gradients are smaller: multiplying the loss by a large
    scale shifts every gradient of the backward pass up by the same factor, and dividing the weights' gradients by it
    afterwards undoes it in float32. If any gradient overflowed to inf or NaN the step is skipped and the scale halved;
    after growth_interval clean steps in a row it is doubled again."""

    def __init__(self, c: Config):
        self.c, self.scale, self.clean = c, c.init_scale, 0

    def unscale(self, params):
        grads = [p.grad for p in params if p.grad is not None]
        for g in grads:
            g.div_(self.scale)
        return bool(torch.stack([~torch.isfinite(g).all() for g in grads]).any())

    def update(self, found_inf):
        if found_inf:
            self.scale *= self.c.backoff_factor
            self.clean = 0
        else:
            self.clean += 1
            if self.clean == self.c.growth_interval:
                self.scale *= self.c.growth_factor
                self.clean = 0


# %% [train-step] One mixed-precision training step
def train_step(model, x, y, optimizer, mode, scaler=None):
    """The weights the optimizer updates are float32 in every mode: autocast makes bfloat16 or float16 copies for
    each matmul on the fly, and FP8Linear rounds its inputs itself (the model's linears call fp8_linear)."""
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)
    with torch.autocast(x.device.type, dtype=dtype, enabled=dtype is not None):
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
    optimizer.zero_grad(set_to_none=True)
    if mode == "fp16":
        (loss * scaler.scale).backward()
        found_inf = scaler.unscale(model.parameters())
        if not found_inf:
            optimizer.step()
        scaler.update(found_inf)
    else:
        loss.backward()
        optimizer.step()
    return loss.detach()
`;export{e as default};
