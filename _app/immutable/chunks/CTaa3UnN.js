const e=`# LoRA (low-rank adaptation) on SmolLM2-135M-Instruct, the Post-training Explainer's LoRA tab, as plain PyTorch.
# Every weight of the model stays frozen; each adapted projection W (out × in) gets two small trainable matrices,
# A (r × in) and B (out × r), and computes W x + (α / r) · B A x. B starts at zero, so training starts from the
# model's own outputs, and after training B A can be merged into W, leaving a plain nn.Linear again.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    rank: int = 8  # r
    alpha: int = 16  # α; the update is scaled by α / r = 2
    targets: tuple = ("q_proj", "v_proj")  # peft's default for Llama models, as in the LoRA paper
    d_model: int = 576  # SmolLM2-135M: q_proj is 576 → 576, v_proj 576 → 192 (3 KV heads × 64)
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


# %% [lora-linear] LoRA linear layer
class LoRALinear(nn.Module):
    """h = W x + (α/r) · B (A x): a frozen nn.Linear plus a rank-r update that costs r · (in + out) weights."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base.requires_grad_(False)  # W (and its bias, if any) never change
        self.A = nn.Parameter(torch.empty(rank, base.in_features))  # (r, in)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))  # (out, r): zero, so B A = 0 at the start
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # the same init as nn.Linear's weight
        self.scale = alpha / rank

    def forward(self, x):  # x: (..., in)
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T  # (..., out); never forms the out × in B A


# %% [inject] Adapters on the target projections
def add_adapters(model: nn.Module, c: Config):
    """Freeze the whole model, then wrap every Linear whose name ends in a target (q_proj, v_proj) in LoRALinear."""
    model.requires_grad_(False)
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and name.split(".")[-1] in c.targets]
    for name in names:
        parent, _, child = name.rpartition(".")
        owner = model.get_submodule(parent)
        setattr(owner, child, LoRALinear(getattr(owner, child), c.rank, c.alpha))
    return names


def count_parameters(model: nn.Module):
    """(trainable, total): for SmolLM2-135M with r = 8 on q and v, 30 · 8 · (576 + 576 + 576 + 192) = 460,800."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


# %% [lora-step] A training step
class AdamW:
    """AdamW over the adapters only; the optimizer keeps two moments per trainable weight, so its state is tiny too."""

    def __init__(self, params, c: Config):
        self.params, self.c, self.t = [p for p in params if p.requires_grad], c, 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        (b1, b2), lr = self.c.betas, self.c.lr
        for p, m, v in zip(self.params, self.m, self.v):
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            p.sub_(lr * (m / (1 - b1**self.t)) / ((v / (1 - b2**self.t)).sqrt() + self.c.eps))


def lora_step(model, optimizer, loss_fn):
    """One step: only A and B get gradients. While B = 0, A's gradient Bᵀ(∂L/∂h)xᵀ is zero too, so the first
    step moves B alone."""
    for p in optimizer.params:
        p.grad = None
    loss = loss_fn(model)
    loss.backward()
    optimizer.step()
    return loss.detach()


# %% [merge] Merging back into W
@torch.no_grad()
def merge(layer: LoRALinear) -> nn.Linear:
    """W' = W + (α/r) · B A: one plain Linear that computes exactly what the adapted layer computed."""
    merged = nn.Linear(layer.base.in_features, layer.base.out_features, bias=layer.base.bias is not None)
    merged.weight.copy_(layer.base.weight + layer.scale * layer.B @ layer.A)  # (out, in)
    if layer.base.bias is not None:
        merged.bias.copy_(layer.base.bias)
    return merged


def merge_all(model: nn.Module):
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, merge(module))
    return model
`;export{e as default};
