const e=`# t-SNE and UMAP, as the Classical ML Explainer's t-SNE tab runs them on the digits: t-SNE turns distances into
# neighbour probabilities P (a Gaussian per point, its width set by the perplexity), places points in 2-D where a
# Student-t gives probabilities Q, and moves them down the gradient of KL(P‖Q) with momentum and per-coordinate gains;
# UMAP builds a fuzzy k-nearest-neighbour graph and lays it out with a curve fitted to min_dist. This is a PyTorch
# reimplementation of scikit-learn's exact t-SNE and umap-learn's graph; the check compares it with those libraries
# and the recorded run.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    perplexity: float = 30.0  # about how many neighbours each point's Gaussian covers
    early_exaggeration: float = 12.0  # P is multiplied by this for the first 250 iterations
    exploration: int = 250
    iterations: int = 1000
    learning_rate: float = 50.0  # scikit-learn's 'auto': max(n / exaggeration / 4, 50)
    min_gain: float = 0.01
    n_neighbors: int = 15  # UMAP
    min_dist: float = 0.1  # UMAP: how tightly the layout may pack points


# %% [affinities] Neighbour probabilities P
def conditional_p(sqdist, perplexity, steps=100, tolerance=1e-5):
    """For each point i, a Gaussian over the others, p_j|i ∝ exp(−β_i d_ij²), with β_i found by bisection so that the
    entropy of p_·|i is log(perplexity): a point in a dense region gets a narrow Gaussian, an isolated one a wide one."""
    n = len(sqdist)
    off = ~torch.eye(n, dtype=torch.bool)
    beta = torch.ones(n, dtype=sqdist.dtype)
    lo = torch.full((n,), -torch.inf, dtype=sqdist.dtype)
    hi = torch.full((n,), torch.inf, dtype=sqdist.dtype)
    active = torch.ones(n, dtype=torch.bool)
    P = torch.zeros_like(sqdist)
    target = torch.log(torch.tensor(perplexity, dtype=sqdist.dtype))
    for _ in range(steps):
        p = torch.exp(-sqdist[active] * beta[active, None]) * off[active]
        s = p.sum(1).clamp_min(1e-8)
        p = p / s[:, None]
        entropy = torch.log(s) + beta[active] * (sqdist[active] * p).sum(1)
        P[active] = p
        diff = entropy - target
        idx = active.nonzero()[:, 0]
        done = diff.abs() <= tolerance
        up, down = ~done & (diff > 0), ~done & (diff <= 0)  # too flat: narrow it; too peaked: widen it
        b = beta[idx]
        lo[idx[up]] = b[up]
        beta[idx[up]] = torch.where(torch.isinf(hi[idx[up]]), b[up] * 2, (b[up] + hi[idx[up]]) / 2)
        hi[idx[down]] = b[down]
        beta[idx[down]] = torch.where(torch.isinf(lo[idx[down]]), b[down] / 2, (b[down] + lo[idx[down]]) / 2)
        active[idx[done]] = False
        if not active.any():
            break
    return P, beta


def joint_p(X, perplexity):
    """p_ij = (p_j|i + p_i|j) / Σ, symmetric and summing to 1 over all pairs (scikit-learn rounds d² to float32)."""
    sqdist = torch.cdist(X, X).square().float().double()
    P, _ = conditional_p(sqdist, perplexity)
    P = P + P.T
    return (P / P.sum()).clamp_min(torch.finfo(torch.float64).eps)


# %% [kl-gradient] Q and the gradient of KL(P‖Q)
def kl_gradient(Y, P):
    """q_ij ∝ (1 + |y_i − y_j|²)⁻¹, a Student-t with one degree of freedom, whose heavy tail lets far-apart clusters
    sit far apart. KL(P‖Q) = Σ p_ij log(p_ij / q_ij), and ∂KL/∂y_i = 4 Σ_j (p_ij − q_ij)(1 + |y_i − y_j|²)⁻¹ (y_i − y_j):
    springs that pull neighbours in P together and push everything else apart."""
    n = len(Y)
    w = 1 / (1 + torch.cdist(Y, Y).square())
    w = w * ~torch.eye(n, dtype=torch.bool)
    Q = (w / w.sum()).clamp_min(torch.finfo(torch.float64).eps)
    off = ~torch.eye(n, dtype=torch.bool)
    kl = (P[off] * torch.log(P[off] / Q[off])).sum()
    PQ = (P - Q) * w * off
    grad = 4 * (PQ.sum(1, keepdim=True) * Y - PQ @ Y)
    return kl, grad


# %% [optimizer] Momentum and gains
def step(Y, update, gains, grad, momentum, c: Config):
    """scikit-learn's update (after Jacobs' delta-bar-delta): a coordinate's gain grows by 0.2 while its gradient
    keeps the sign that the last update followed and shrinks ×0.8 when it flips; update = momentum·update −
    lr·gain·grad. Momentum is 0.5 while P is exaggerated and 0.8 after, and update and gains restart at the switch."""
    same = update * grad < 0
    gains = torch.where(same, gains + 0.2, gains * 0.8).clamp_min(c.min_gain)
    update = momentum * update - c.learning_rate * gains * grad
    return Y + update, update, gains


def run(Y, update, gains, P, start, iterations, c: Config):
    """\`iterations\` steps from iteration \`start\`, with P exaggerated before c.exploration."""
    for i in range(start, start + iterations):
        if i == c.exploration:
            update, gains = torch.zeros_like(update), torch.ones_like(gains)
        early = i < c.exploration
        _, grad = kl_gradient(Y, P * (c.early_exaggeration if early else 1))
        Y, update, gains = step(Y, update, gains, grad, 0.5 if early else 0.8, c)
    return Y, update, gains


# %% [umap-graph] UMAP's fuzzy neighbour graph
def umap_graph(X, c: Config, steps=64, tolerance=1e-5):
    """For each point: its k nearest neighbours (itself included, as umap-learn counts them), ρ = distance to the
    nearest other point, and σ by bisection so that Σ_j exp(−(d_ij − ρ)/σ) = log₂ k. Edge weights
    w_ij = exp(−(d_ij − ρ_i)/σ_i), so every point's nearest neighbour gets weight 1 however far it is; then made
    symmetric by the fuzzy union w + wᵀ − w∘wᵀ."""
    n, k = len(X), c.n_neighbors
    D = torch.cdist(X, X)
    order = torch.sort(D, dim=1, stable=True).indices[:, :k]  # ties keep index order, as umap-learn's mergesort
    dist = D.gather(1, order)
    rho = torch.where(dist > 0, dist, torch.inf).min(1).values
    target = torch.log2(torch.tensor(float(k), dtype=X.dtype))
    lo, hi = torch.zeros(n, dtype=X.dtype), torch.full((n,), torch.inf, dtype=X.dtype)
    sigma = torch.ones(n, dtype=X.dtype)
    active = torch.ones(n, dtype=torch.bool)
    for _ in range(steps):
        psum = torch.exp(-(dist[:, 1:] - rho[:, None]).clamp_min(0) / sigma[:, None]).sum(1)
        active &= (psum - target).abs() >= tolerance  # a point stops when its sum is close enough
        big, small = active & (psum > target), active & (psum <= target)
        hi = torch.where(big, sigma, hi)
        lo = torch.where(small, sigma, lo)
        sigma = torch.where(big | (small & ~torch.isinf(hi)), (lo + hi) / 2, torch.where(small, sigma * 2, sigma))
    sigma = torch.maximum(sigma, 1e-3 * dist.mean(1))  # umap-learn's floor for very dense points
    weights = torch.exp(-(dist - rho[:, None]).clamp_min(0) / sigma[:, None])
    weights[order == torch.arange(n)[:, None]] = 0  # no edge to itself
    W = torch.zeros(n, n, dtype=X.dtype).scatter_(1, order, weights)
    return W + W.T - W * W.T, sigma, rho


# %% [umap-curve] UMAP's low-dimensional similarity curve
def umap_curve(min_dist, spread=1.0):
    """The layout scores a pair at distance d by 1 / (1 + a d^{2b}); a and b are fitted (least squares, 300 points
    on [0, 3·spread]) to a curve that is 1 up to min_dist and falls as exp(−(d − min_dist)/spread) after it."""
    d = torch.linspace(0, 3 * spread, 300, dtype=torch.float64)
    target = torch.where(d < min_dist, torch.ones_like(d), torch.exp(-(d - min_dist) / spread))
    params = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([params], max_iter=500, tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        a, b = params
        loss = ((1 / (1 + a * d ** (2 * b)) - target) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return params.detach()
`;export{e as default};
