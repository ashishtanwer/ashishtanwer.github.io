const e=`# Wanda (arXiv:2306.11695) and SparseGPT (arXiv:2301.00774) side by side on one linear layer of Qwen2.5-0.5B-Instruct.
# Both start from Optimal Brain Surgeon's cost of dropping weight w_ij, w² / [H⁻¹]_jj with H = 2XXᵀ/n from calibration
# inputs X. Wanda keeps only H's diagonal and compares weights within each output row, with no update; SparseGPT uses
# the full inverse through its Cholesky factor, chooses the mask as it sweeps the columns, and updates the kept weights.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    hidden: int = 896  # input width of k_proj (and q, v, gate, up)
    kv_width: int = 128  # rows of k_proj: 2 key-value heads × 64
    sparsity: float = 0.5
    n: int = 2  # 2:4 sparsity: prune n of every m consecutive inputs of a row
    m: int = 4
    blocksize: int = 128  # SparseGPT: columns whose 50 % mask is chosen together
    percdamp: float = 0.01  # SparseGPT: damping added to H's diagonal, as a fraction of its mean


# %% [weighting] What each method weighs an input channel by
def inverse_factor(H, percdamp=0.01):
    """Upper Cholesky factor U of (H + λI)⁻¹, as SparseGPT computes it (UᵀU = (H + λI)⁻¹)."""
    H = H.clone()
    dead = torch.diag(H) == 0  # inputs that were always zero
    H[dead, dead] = 1
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(H.shape[0])
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)


def wanda_weight(H):
    # √H_jj = √2 ‖X_j‖ / √n: input j's size alone, as if H were diagonal
    return torch.diag(H).sqrt()


def sparsegpt_weight(H, percdamp=0.01):
    """1 / U_jj, where 1 / U_jj² = H_jj − H_jF H_FF⁻¹ H_Fj (with λ on the diagonal) over the inputs F after j: the
    part of input j that those inputs cannot reproduce, so cannot take over when j's weight is dropped."""
    return 1 / torch.diag(inverse_factor(H, percdamp))


# %% [score] Two scores for one weight
def wanda_score(weight, scaler_row):
    # |W_ij| · ‖X_j‖ (scaler_row = ‖X_j‖² / n); its square is w² H_jj up to a constant
    return weight.abs() * scaler_row.sqrt()[None, :]


def obs_score(weight, U):
    # OBS: the output error of dropping w_ij alone, w² / [H⁻¹]_jj, with SparseGPT's U_jj² for [H⁻¹]_jj
    return weight**2 / torch.diag(U)[None, :] ** 2


# %% [mask] Wanda's mask: within each output row
def wanda_mask(score, sparsity=0.5, n=0, m=0):
    """True where a weight is pruned: the lowest half of every row, or the n lowest of every m."""
    if n:
        groups = score.reshape(score.shape[0], -1, m)
        lowest = groups.topk(n, dim=-1, largest=False).indices
        return torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, lowest, True).reshape(score.shape)
    order = score.sort(dim=-1, stable=True).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(1, order[:, : int(score.shape[1] * sparsity)], True)


# %% [update] SparseGPT's sweep: its own mask, or the update alone for a given one
@torch.no_grad()
def obs_sweep(weight, H, pruned=None, sparsity=0.5, n=0, m=0, blocksize=128, percdamp=0.01):
    """Column by column, each dropped weight's error is pushed onto the columns not yet visited, so the kept weights
    make up for it. Without \`pruned\` SparseGPT picks the mask as it goes (50 %: the lowest OBS scores of each block,
    across all its rows; n:m: the n lowest of each group, from the already updated weights). Returns the pruned
    weights and the score every weight was judged by."""
    W = weight.clone().float()
    W[:, torch.diag(H) == 0] = 0
    U = inverse_factor(H, percdamp)
    scores = torch.zeros_like(W)
    for start in range(0, W.shape[1], blocksize):
        end = min(start + blocksize, W.shape[1])
        block, U1 = W[:, start:end].clone(), U[start:end, start:end]
        final, errors = torch.zeros_like(block), torch.zeros_like(block)
        if pruned is not None:
            drop = pruned[:, start:end].clone()
        elif n:
            drop = torch.zeros_like(block, dtype=torch.bool)
        else:
            scores[:, start:end] = obs_score(block, U1)
            drop = scores[:, start:end] <= scores[:, start:end].flatten().sort()[0][int(block.numel() * sparsity)]
        for i in range(end - start):
            if pruned is None and n and i % m == 0:
                group = obs_score(block[:, i:i + m], U1[i:i + m, i:i + m])
                scores[:, start + i:start + i + m] = group
                drop.scatter_(1, i + group.topk(n, dim=1, largest=False)[1], True)
            w = block[:, i]
            q = torch.where(drop[:, i], torch.zeros_like(w), w)
            final[:, i] = q
            error = (w - q) / U1[i, i]
            block[:, i:] -= error.unsqueeze(1).matmul(U1[i, i:].unsqueeze(0))
            errors[:, i] = error
        W[:, start:end] = final
        W[:, end:] -= errors.matmul(U[start:end, end:])
    return W, scores


# %% [error] How much the layer's output moves
def output_error(weight, pruned, hessian):
    """‖(W − Ŵ)X‖² / ‖WX‖² over the calibration inputs X, from H = 2XXᵀ/n (the scale cancels)."""
    delta = (weight - pruned).double()
    weight, hessian = weight.double(), hessian.double()
    return ((delta @ hessian) * delta).sum() / ((weight @ hessian) * weight).sum()
`;export{e as default};
