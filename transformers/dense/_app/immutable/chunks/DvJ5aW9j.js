const e=`# Gaussian process regression, as the Classical ML Explainer's Gaussian-process tab fits the Mauna Loa CO2 record: a
# kernel that says how similar two times' values are, the posterior over functions given noisy observations
# (Cholesky), samples from it, and the log marginal likelihood that picks the kernel's hyperparameters. This is a
# PyTorch reimplementation; the tab's numbers come from scikit-learn's GaussianProcessRegressor, and the check
# compares these blocks with that recording.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    amplitude: float = 3.434  # σ_f, ppm: how far the function strays from its mean
    length: float = 0.2107  # ℓ, years: how far apart two times can be and still move together
    noise: float = 0.2389  # σ_n, ppm: measurement noise on each monthly mean
    jitter: float = 1e-8  # added to a posterior covariance (× σ_f²) before its Cholesky factor


# %% [kernel] The kernel: how values at two times covary
def rbf(a, b, amplitude, length):
    """k(t, t′) = σ_f² exp(−(t − t′)² / 2ℓ²) for every pair of a (n,) and b (m,)."""
    return amplitude**2 * torch.exp(-0.5 * (a[:, None] - b[None]) ** 2 / length**2)


def periodic(a, b, length, period):
    """scikit-learn's ExpSineSquared: exp(−2 sin²(π|t − t′|/p) / ℓ²), 1 whenever t − t′ is a whole number of periods."""
    return torch.exp(-2 * torch.sin(torch.pi * (a[:, None] - b[None]).abs() / period) ** 2 / length**2)


# %% [posterior] The posterior over functions
def posterior(t, y, grid, c: Config):
    """Given y = f(t) + noise, f at the grid points is Gaussian with mean K*ᵀ(K + σ_n²I)⁻¹y and covariance
    K** − K*ᵀ(K + σ_n²I)⁻¹K*, computed with one Cholesky factor. With no data it is the prior: mean 0, cov K**."""
    K_ss = rbf(grid, grid, c.amplitude, c.length)
    if len(t) == 0:
        return torch.zeros(len(grid), dtype=grid.dtype), K_ss
    K = rbf(t, t, c.amplitude, c.length) + c.noise**2 * torch.eye(len(t), dtype=t.dtype)
    K_s = rbf(t, grid, c.amplitude, c.length)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y[:, None], L)[:, 0]
    v = torch.linalg.solve_triangular(L, K_s, upper=False)
    return K_s.T @ alpha, K_ss - v.T @ v


# %% [samples] Functions drawn from it
def samples(mean, cov, z, c: Config):
    """mean + L z with L the Cholesky factor of cov (plus a little jitter): the same z for every posterior, so a
    sample only moves because the data or the kernel did."""
    L = torch.linalg.cholesky(cov + c.jitter * c.amplitude**2 * torch.eye(len(cov), dtype=cov.dtype))
    return (mean[:, None] + L @ z).T


# %% [marginal-likelihood] The log marginal likelihood
def log_marginal_likelihood(theta, t, y):
    """log p(y | θ) = −½ yᵀK⁻¹y − ½ log|K| − n/2 log 2π, with K = σ_f²·RBF(ℓ) + σ_n²I and θ = log(σ_f², ℓ, σ_n²),
    scikit-learn's parameters. The first term rewards fitting the data, the second penalizes a flexible kernel."""
    amplitude2, length, noise2 = torch.exp(theta)
    K = amplitude2 * torch.exp(-0.5 * (t[:, None] - t[None]) ** 2 / length**2) + noise2 * torch.eye(len(t), dtype=t.dtype)
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve(y[:, None], L)[:, 0]
    return -0.5 * y @ alpha - torch.log(torch.diagonal(L)).sum() - 0.5 * len(t) * torch.log(torch.tensor(2 * torch.pi))


def gradient(theta, t, y):
    """∂ log p(y | θ) / ∂θ, by autograd."""
    theta = theta.clone().requires_grad_(True)
    value = log_marginal_likelihood(theta, t, y)
    (grad,) = torch.autograd.grad(value, theta)
    return value.detach(), grad


# %% [fit] Hyperparameters by gradient ascent
def fit(theta0, t, y, steps=200):
    """Maximize log p(y | θ) from a start θ₀ with L-BFGS. Different starts can end in different local optima:
    short ℓ with little noise, long ℓ with a lot."""
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=steps, tolerance_grad=1e-10, tolerance_change=1e-14,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = -log_marginal_likelihood(theta, t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach()
`;export{e as default};
