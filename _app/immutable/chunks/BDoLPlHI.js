const e=`# TIES merging (Yadav et al., 2023, "TIES-Merging: Resolving Interference When Merging Models") on the Post-training
# Explainer's TIES tab, as plain PyTorch: Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct merged onto their
# shared base, Qwen2.5-0.5B. Every tensor of the state dict is merged on its own: each fine-tune's task vector is
# trimmed to its largest entries, a sign is elected per entry, and the entries that agree with it are averaged.
# As mergekit's \`ties\` method (sum consensus, magnitude sparsification, normalize=True).
from dataclasses import dataclass

import torch


@dataclass
class Config:
    base: str = "Qwen/Qwen2.5-0.5B"
    finetuned: tuple = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    density: float = 0.2  # share of each task vector's entries kept, per tensor
    lam: float = 1.0  # λ, the scale of the merged task vector


# %% [task-vector] Task vectors
def task_vector(finetuned: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """τ = θ_ft − θ_base: what fine-tuning changed in one tensor, entry by entry."""
    return finetuned.float() - base.float()


# %% [trim] Trim to the top k%
def trim(tau: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the k = ⌊density · n⌋ entries of largest |τ| and zero the rest. Entries of equal |τ| are taken in index
    order (a stable sort), so exactly k survive."""
    if density >= 1:
        return tau
    k = int(density * tau.numel())
    order = torch.argsort(tau.abs().flatten(), descending=True, stable=True)
    keep = torch.zeros(tau.numel(), dtype=tau.dtype)
    keep[order[:k]] = 1
    return tau * keep.view_as(tau)


# %% [elect] Elect a sign
def elect(trimmed: torch.Tensor) -> torch.Tensor:
    """γ = sign(Σ_t τ̂_t) per entry: the sign carrying more total mass across the trimmed task vectors
    (trimmed: (models, ...)). A sum of exactly 0 elects +."""
    return torch.where(trimmed.sum(0) >= 0, 1.0, -1.0)


# %% [disjoint-mean] Disjoint mean
def disjoint_mean(trimmed: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """τ_m = mean of the τ̂_t whose sign is γ, per entry. Zeros (trimmed away) and entries of the other sign are left
    out of both the sum and the count; an entry no model keeps stays 0."""
    agree = (trimmed.sign() == sign).to(trimmed.dtype)
    return (trimmed * agree).sum(0) / agree.sum(0).clamp_min(1)


# %% [merge] The merged model
@torch.no_grad()
def ties_merge(base: dict, finetuned: list[dict], density: float, lam: float = 1.0) -> dict:
    """θ = θ_base + λ · τ_m, tensor by tensor, for every tensor of the base's state dict."""
    merged = {}
    for name, b in base.items():
        trimmed = torch.stack([trim(task_vector(ft[name], b), density) for ft in finetuned])
        merged[name] = b.float() + lam * disjoint_mean(trimmed, elect(trimmed))
    return merged


def task_arithmetic(base: dict, finetuned: list[dict], lam: float = 1.0) -> dict:
    """The baseline TIES improves on: θ_base + λ · Σ_t τ_t, nothing trimmed and no signs elected
    (λ = 1/T is the plain average of the fine-tunes)."""
    return {name: b.float() + lam * sum(task_vector(ft[name], b) for ft in finetuned) for name, b in base.items()}
`;export{e as default};
