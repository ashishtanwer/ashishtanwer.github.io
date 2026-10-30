const e=`# DoRA (Liu et al., 2024, weight-decomposed low-rank adaptation) on SmolLM2-135M-Instruct, the Post-training
# Explainer's DoRA tab, as plain PyTorch. Each adapted weight W (out × in) is split into a magnitude, one number per
# output (the norm of that row of W), and a direction, W divided by those norms. LoRA's B A updates the direction;
# the magnitudes are trained directly. As in peft's use_dora=True.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    rank: int = 8
    alpha: int = 16
    targets: tuple = ("q_proj", "v_proj")
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


# %% [decompose] Magnitude and direction
def decompose(W: torch.Tensor):
    """W = m · (W / m): m holds the length of each output's weight row (the paper's column-wise norm of Wᵀ), the
    direction holds unit rows."""
    m = torch.linalg.norm(W, dim=1)  # (out,)
    return m, W / m[:, None]


# %% [dora-linear] The DoRA layer
class DoRALinear(nn.Module):
    """h = m ⊙ (V x) / ‖V‖ with V = W + (α/r)·B A: LoRA moves the direction, m sets each output's length.
    ‖V‖ is treated as a constant in the backward pass (the paper's §4.3; peft does the same)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.m = nn.Parameter(decompose(base.weight.detach())[0].clone())  # starts at ‖W‖: the first output is W x

    def combined(self):
        return self.base.weight + self.scale * self.B @ self.A  # V, (out, in)

    def forward(self, x):
        V = self.combined()
        norm = torch.linalg.norm(V, dim=1).detach()  # (out,), no gradient through the norm
        return (self.m / norm) * F.linear(x, V) + (0 if self.base.bias is None else self.base.bias)


# %% [dora-step] A training step
def dora_step(model, optimizer, loss_fn):
    """A, B and m all get gradients and AdamW updates (m adds one weight per output of each adapted layer)."""
    optimizer.zero_grad()
    loss = loss_fn(model)
    loss.backward()
    optimizer.step()
    return loss.detach()


# %% [merge] Merging back
@torch.no_grad()
def merge(layer: DoRALinear) -> nn.Linear:
    """W′ = m ⊙ V / ‖V‖, row by row: one plain Linear computing exactly what the DoRA layer computed."""
    V = layer.combined()
    merged = nn.Linear(V.shape[1], V.shape[0], bias=layer.base.bias is not None)
    merged.weight.copy_((layer.m / torch.linalg.norm(V, dim=1))[:, None] * V)
    if layer.base.bias is not None:
        merged.bias.copy_(layer.base.bias)
    return merged


def change(W0: torch.Tensor, W: torch.Tensor):
    """The paper's analysis: ΔM, the mean change of the row lengths, and ΔD, the mean 1 − cos between old and new
    rows, from W0 to W (W0 + (α/r)·B A for LoRA, the merged W′ for DoRA)."""
    delta_m = (torch.linalg.norm(W, dim=1) - torch.linalg.norm(W0, dim=1)).abs().mean()
    delta_d = (1 - F.cosine_similarity(W, W0, dim=1)).mean()
    return delta_m, delta_d
`;export{e as default};
