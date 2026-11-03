const n=`# The Adapters family compared (LoRA, QLoRA and DoRA on SmolLM2-135M-Instruct, the Post-training Explainer's Adapters
# tab), as plain PyTorch: what each method adds to a frozen weight W₀ (out × in), how to read its singular values
# without forming the out × in matrix, how alike two updates are, what QLoRA's 4-bit base gets wrong, and what each
# method trains and keeps in AdamW. The adapters themselves are the LoRA, QLoRA and DoRA tabs' blocks.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    rank: int = 8
    alpha: int = 16
    targets: tuple = ("q_proj", "v_proj")


# %% [spectrum] The singular values of a low-rank update
def low_rank_svd(A: torch.Tensor, B: torch.Tensor, scale: float):
    """(α/r)·B·A has rank ≤ r. With B = Q_B R_B and Aᵀ = Q_A R_A (QR), it equals Q_B · (α/r)·R_B R_Aᵀ · Q_Aᵀ, so the
    r singular values are those of the r × r core and the left singular vectors are Q_B times the core's."""
    QB, RB = torch.linalg.qr(B.double())  # (out, r), (r, r)
    QA, RA = torch.linalg.qr(A.double().T)  # (in, r), (r, r)
    U, S, Vh = torch.linalg.svd(scale * RB @ RA.T)  # r × r
    return S, QB @ U, QA @ Vh.T  # (r,), (out, r), (in, r)


# %% [dora-update] DoRA's update to W₀
def dora_update(W0: torch.Tensor, A: torch.Tensor, B: torch.Tensor, m: torch.Tensor, scale: float):
    """DoRA's merged weight is m ⊙ V / ‖V‖ with V = W₀ + (α/r)·B·A (row norms). Its change from W₀ splits into a
    magnitude part, each row of W₀ stretched by f − 1 with f = m / ‖V‖, and a direction part, the LoRA update scaled
    by f. The first is a full-rank matrix, so DoRA's merged update is not rank r even though B·A is."""
    low = scale * B @ A
    f = m / torch.linalg.norm(W0 + low, dim=1)  # (out,), 1 before the first step
    magnitude = (f - 1)[:, None] * W0
    direction = f[:, None] * low
    return magnitude + direction, magnitude, direction


# %% [similarity] How alike two updates are
def cosine(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """⟨X, Y⟩ / (‖X‖ ‖Y‖) over all entries: 1 for the same change up to scale, 0 for unrelated ones."""
    return (X * Y).sum() / (X.norm() * Y.norm())


def subspace_overlap(Ua: torch.Tensor, Ub: torch.Tensor) -> torch.Tensor:
    """‖Uaᵀ Ub‖² / r for two sets of r orthonormal columns (the outputs each update writes to): the mean squared
    cosine of their principal angles, 1 when they span the same subspace, about r / out for random ones."""
    return (Ua.T @ Ub).pow(2).sum() / Ua.shape[1]


# %% [base-error] What the 4-bit base gets wrong
def base_error(W0: torch.Tensor, D: torch.Tensor, update: torch.Tensor, rank: int):
    """QLoRA's forward pass uses D, W₀ dequantized from NF4, not W₀. Returns ‖D − W₀‖ / ‖W₀‖, the share of
    ‖D − W₀‖² the best rank-r matrix could cancel (its r largest squared singular values, Eckart–Young), and the
    cosine between the trained update and W₀ − D: how much the adapter moved back towards the float weight."""
    E = (D - W0).double()
    s = torch.linalg.svdvals(E)
    return E.norm() / W0.double().norm(), s[:rank].pow(2).sum() / s.pow(2).sum(), cosine(update.double(), -E)


# %% [cost] What each method trains and keeps
def trainable(shapes: dict, rank: int, dora: bool) -> int:
    """A (r × in) and B (out × r) per adapted matrix; DoRA adds its magnitudes m, one per output row."""
    return sum(rank * (n_in + n_out) + (n_out if dora else 0) for n_out, n_in in shapes.values())


def training_bytes(n_trainable: int, n_tensors: int, base_bytes: int) -> dict:
    """float32 weights, their gradients and AdamW's two moments (exp_avg, exp_avg_sq) plus one float32 step count per
    trainable tensor, beside the frozen base as it is held (4 bytes a weight in float32, about 0.5 in NF4 plus its
    absmax constants)."""
    return {"base": base_bytes, "adapters": 4 * n_trainable, "grads": 4 * n_trainable,
            "adam": 8 * n_trainable + 4 * n_tensors}
`;export{n as default};
