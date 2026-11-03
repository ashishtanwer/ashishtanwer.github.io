const e=`# A soft-margin support vector machine, as the Classical ML Explainer's SVM tab fits it: kernels, the polynomial
# kernel's explicit feature map, the dual problem solved by SMO, and the decision function read back from the support
# vectors. This is a PyTorch reimplementation; the tab's numbers come from scikit-learn's SVC (libsvm), and the check
# compares these blocks with that recording.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    C: float = 1.0  # the cost of a unit of slack: how hard a point inside the margin pulls
    degree: int = 2  # polynomial kernel (γ x·z + r)^degree
    gamma: float = 1.0  # polynomial and RBF γ
    coef0: float = 1.0  # polynomial r
    tolerance: float = 1e-10  # stop SMO when the most violating pair violates by less than this


# %% [kernels] Kernels: similarity without features
def poly_kernel(A, B, c: Config):
    """K(x, z) = (γ x·z + r)^d for every row pair of A (n, 2) and B (m, 2)."""
    return (c.gamma * A @ B.T + c.coef0) ** c.degree


def rbf_kernel(A, B, c: Config):
    """K(x, z) = exp(−γ |x − z|²): 1 for the same point, near 0 a few length scales away."""
    return torch.exp(-c.gamma * torch.cdist(A, B) ** 2)


# %% [feature-map] The explicit lift behind the polynomial kernel
def poly_features(X, c: Config):
    """φ(x) for 2-D x and degree 2, so that φ(x)·φ(z) = (γ x·z + r)²: a constant, the two coordinates scaled by
    √(2γr), and the three quadratic monomials. The kernel computes this dot product without forming φ."""
    x1, x2 = X[:, 0], X[:, 1]
    s = (2 * c.gamma * c.coef0) ** 0.5
    return torch.stack([torch.full_like(x1, c.coef0), s * x1, s * x2,
                        c.gamma * x1**2, 2**0.5 * c.gamma * x1 * x2, c.gamma * x2**2], 1)


def weight_vector(alpha_y, X, c: Config):
    """w = Σᵢ αᵢyᵢ φ(xᵢ): the normal of the separating plane in the lifted space (6 numbers for degree 2)."""
    return alpha_y @ poly_features(X, c)


# %% [dual] The dual problem, solved by SMO
def smo(K, y, c: Config, max_steps=200_000):
    """min ½ αᵀQα − Σα with Qᵢⱼ = yᵢyⱼKᵢⱼ, 0 ≤ αᵢ ≤ C and Σ yᵢαᵢ = 0. Each step picks the most violating pair
    (i, j), moves αᵢ and αⱼ along the line that keeps Σ yα fixed to the best point inside the box, and updates the
    gradient G = Qα − 1. y is ±1. Returns αᵢyᵢ and b, as libsvm's dual_coef_ and intercept_."""
    n = len(y)
    alpha = torch.zeros(n, dtype=K.dtype)
    G = -torch.ones(n, dtype=K.dtype)
    for _ in range(max_steps):
        score = -y * G
        up = ((y > 0) & (alpha < c.C)) | ((y < 0) & (alpha > 0))  # may move so that yα grows
        low = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < c.C))  # may move so that yα shrinks
        i = torch.where(up, score, -torch.inf).argmax()
        j = torch.where(low, score, torch.inf).argmin()
        gap = score[i] - score[j]
        if gap < c.tolerance:
            break
        curvature = (K[i, i] + K[j, j] - 2 * K[i, j]).clamp_min(1e-12)
        t = gap / curvature  # αᵢ += yᵢ t and αⱼ −= yⱼ t, clipped so both stay in [0, C]
        t = torch.minimum(t, c.C - alpha[i] if y[i] > 0 else alpha[i])
        t = torch.minimum(t, alpha[j] if y[j] > 0 else c.C - alpha[j])
        alpha[i] += y[i] * t
        alpha[j] -= y[j] * t
        G += y * (K[:, i] - K[:, j]) * t
    free = (alpha > 1e-12) & (alpha < c.C - 1e-12)
    yG = y * G
    if free.any():  # on the margin: yᵢ f(xᵢ) = 1 exactly
        b = -yG[free].mean()
    else:
        b = -(yG[up].min() + yG[low].max()) / 2
    return alpha * y, b


# %% [decision] The decision function and the margin
def decision(alpha_y, b, X_train, X, kernel, c: Config):
    """f(x) = Σᵢ αᵢyᵢ K(xᵢ, x) + b = w·φ(x) + b. Only support vectors (αᵢ > 0) contribute; f = 0 is the boundary
    and f = ±1 the edges of the margin."""
    return kernel(X, X_train, c) @ alpha_y + b


def margin_width(alpha_y, K):
    """2 / |w|, with |w|² = Σᵢⱼ αᵢyᵢ αⱼyⱼ K(xᵢ, xⱼ): the gap between the planes w·φ + b = ±1 in the lifted space."""
    return 2 / torch.sqrt(alpha_y @ K @ alpha_y)


# %% [random-features] An explicit map for the RBF kernel
def random_fourier_features(X, omega, beta):
    """z(x) = √(2/D) cos(Ωᵀx + β) with Ω ~ N(0, 2γ I) and β ~ U[0, 2π): z(x)·z(x′) ≈ exp(−γ|x − x′|²), closer as
    D grows (Rahimi and Recht). The RBF kernel's own feature space has infinitely many dimensions."""
    D = omega.shape[1]
    return (2 / D) ** 0.5 * torch.cos(X @ omega + beta)
`;export{e as default};
