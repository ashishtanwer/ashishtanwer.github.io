const e=`# Expectation–maximization for a Gaussian mixture, as the Classical ML Explainer's EM tab fits the wine data: the
# E-step gives every point a responsibility for each component, the M-step refits each component to the points
# weighted by those responsibilities, and the log-likelihood can only go up. This is a PyTorch reimplementation of
# scikit-learn's GaussianMixture (full covariances); the check compares every recorded iteration.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    components: int = 3
    reg_covar: float = 1e-6  # added to every covariance's diagonal, as scikit-learn does
    tolerance: float = 1e-10  # stop when the average log-likelihood moves less than this
    max_iter: int = 500


# %% [e-step] E-step: responsibilities
def log_gaussian(X, mean, cov):
    """log N(x | μ, Σ) for every row of X (n, d), through the Cholesky factor of Σ."""
    L = torch.linalg.cholesky(cov)
    z = torch.linalg.solve_triangular(L, (X - mean).T, upper=False)
    d = X.shape[1]
    return -0.5 * (z**2).sum(0) - torch.log(torch.diagonal(L)).sum() - 0.5 * d * torch.log(torch.tensor(2 * torch.pi))


def e_step(X, weights, means, covs):
    """r_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j), in log space. Also returns the log-likelihood
    Σ_i log Σ_k π_k N(x_i | μ_k, Σ_k) of these parameters."""
    log_p = torch.stack([torch.log(w) + log_gaussian(X, m, S) for w, m, S in zip(weights, means, covs)], 1)
    log_norm = torch.logsumexp(log_p, 1)
    return torch.exp(log_p - log_norm[:, None]), log_norm.sum()


# %% [m-step] M-step: refit each component
def m_step(X, resp, c: Config):
    """N_k = Σ_i r_ik; π_k = N_k / n; μ_k = Σ_i r_ik x_i / N_k; Σ_k = Σ_i r_ik (x_i − μ_k)(x_i − μ_k)ᵀ / N_k (+ reg).
    A component is refitted to every point at once, each weighted by how much it is responsible for it."""
    Nk = resp.sum(0) + 10 * torch.finfo(resp.dtype).eps
    means = resp.T @ X / Nk[:, None]
    covs = []
    for k in range(len(Nk)):
        diff = X - means[k]
        covs.append((resp[:, k, None] * diff).T @ diff / Nk[k] + c.reg_covar * torch.eye(X.shape[1], dtype=X.dtype))
    return Nk / len(X), means, torch.stack(covs)


# %% [em] Iterate until nothing moves
def em(X, weights, means, covs, c: Config):
    """E-step, M-step, repeat. Returns the parameters after every iteration and the log-likelihood of each, starting
    with the initial ones. The log-likelihood never decreases; where it ends depends on where it started."""
    history = [(weights, means, covs)]
    _, ll = e_step(X, weights, means, covs)
    lls = [ll]
    for _ in range(c.max_iter):
        resp, _ = e_step(X, weights, means, covs)
        weights, means, covs = m_step(X, resp, c)
        _, ll = e_step(X, weights, means, covs)
        history.append((weights, means, covs))
        lls.append(ll)
        if abs(lls[-1] - lls[-2]) / len(X) < c.tolerance:
            break
    return history, torch.stack(lls)
`;export{e as default};
