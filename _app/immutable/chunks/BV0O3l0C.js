const e=`# DARE (Yu et al., 2024, "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch")
# on the Post-training Explainer's DARE tab, as plain PyTorch: Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct
# merged onto their shared base, Qwen2.5-0.5B. Each task vector's entries are dropped at random with probability p and
# the survivors scaled by 1/(1 − p), so every entry keeps its expected value; the sparse deltas are then added to the
# base (DARE + task arithmetic, mergekit's \`dare_linear\`) or merged with TIES's sign election and disjoint mean
# (DARE + TIES, as peft's \`dare_ties\`).
from dataclasses import dataclass

import torch


@dataclass
class Config:
    base: str = "Qwen/Qwen2.5-0.5B"
    finetuned: tuple = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    p: float = 0.9  # drop rate: the share of each task vector's entries set to 0
    lam: float = 1.0  # λ, the scale of the merged task vector
    seed: int = 0


# %% [task-vector] Task vectors
def task_vector(finetuned: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """δ = θ_ft − θ_base: what fine-tuning changed in one tensor, entry by entry."""
    return finetuned.float() - base.float()


# %% [drop] Drop at random
def drop(delta: torch.Tensor, p: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """m ~ Bernoulli(1 − p) per entry, from the seeded generator; returns (m ⊙ δ, m)."""
    mask = torch.bernoulli(torch.full_like(delta, 1 - p), generator=generator)
    return delta * mask, mask


# %% [rescale] Rescale the survivors
def rescale(dropped: torch.Tensor, p: float) -> torch.Tensor:
    """δ̂ = m ⊙ δ / (1 − p): E[δ̂] = δ, so the layer's expected output x·δ̂ᵀ is x·δᵀ."""
    return dropped / (1 - p)


def dare(delta: torch.Tensor, p: float, generator: torch.Generator) -> torch.Tensor:
    """Drop and rescale one task vector (p = 0 returns it unchanged)."""
    return delta if p == 0 else rescale(drop(delta, p, generator)[0], p)


# %% [merge] DARE + task arithmetic
@torch.no_grad()
def dare_merge(base: dict, finetuned: list[dict], p: float, lam: float = 1.0, seed: int = 0) -> dict:
    """θ = θ_base + λ · Σ_t DARE(δ_t), tensor by tensor. One generator, seeded once, draws the masks in the base's
    tensor order, fine-tune after fine-tune. With one fine-tune this is DARE on that model alone."""
    generator = torch.Generator().manual_seed(seed)
    merged = {}
    for name, b in base.items():
        merged[name] = b.float() + lam * sum(dare(task_vector(ft[name], b), p, generator) for ft in finetuned)
    return merged


# %% [dare-ties] DARE + TIES
def elect(deltas: torch.Tensor) -> torch.Tensor:
    """γ = sign(Σ_t δ̂_t) per entry (deltas: (models, ...)); a sum of exactly 0 elects +."""
    return torch.where(deltas.sum(0) >= 0, 1.0, -1.0)


def disjoint_mean(deltas: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Mean of the δ̂_t whose sign is γ, per entry; dropped entries (0) and the other sign are left out."""
    agree = (deltas.sign() == sign).to(deltas.dtype)
    return (deltas * agree).sum(0) / agree.sum(0).clamp_min(1)


@torch.no_grad()
def dare_ties_merge(base: dict, finetuned: list[dict], p: float, lam: float = 1.0, seed: int = 0) -> dict:
    """TIES with its magnitude trim replaced by DARE: drop and rescale each δ_t, elect a sign per entry, average the
    entries that agree with it, θ = θ_base + λ · τ_m. The masks come from the generator as in dare_merge."""
    generator = torch.Generator().manual_seed(seed)
    merged = {}
    for name, b in base.items():
        deltas = torch.stack([dare(task_vector(ft[name], b), p, generator) for ft in finetuned])
        merged[name] = b.float() + lam * disjoint_mean(deltas, elect(deltas))
    return merged
`;export{e as default};
