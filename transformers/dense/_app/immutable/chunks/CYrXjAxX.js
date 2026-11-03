const n=`# Edge of stability, as the Pretraining Explainer records it (Cohen et al. 2021): full-batch gradient descent on 5,000
# CIFAR-10 training images with a small fully connected tanh network and the squared loss, at a few fixed learning
# rates η. Every few steps the sharpness, the largest eigenvalue of the loss's Hessian, is measured by Lanczos on
# Hessian–vector products. On a quadratic, gradient descent is stable only while the sharpness stays below 2/η; the
# runs show the sharpness rising until it reaches 2/η and then hovering there while the loss keeps falling.
# scripts/pretraining/edge_of_stability_runs.py trains with these blocks.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    examples: int = 5000  # the first 5,000 CIFAR-10 training images
    width: int = 200  # two hidden layers of 200 tanh units
    classes: int = 10
    lr: float = 2 / 100  # η; gradient descent is stable on a quadratic while the sharpness is below 2/η = 100
    lanczos_steps: int = 20  # Hessian–vector products per sharpness measurement
    seed: int = 0


# %% [data] Data
def prepare(images, labels, c: Config):
    """uint8 images (n, 32, 32, 3) → one row of 3,072 inputs per image, each colour channel shifted and scaled to mean 0
    and standard deviation 1 over the subset; targets are one-hot rows of 10."""
    x = torch.as_tensor(images[: c.examples]).float().div(255)  # (n, 32, 32, 3)
    mean = x.mean(dim=(0, 1, 2))
    std = x.std(dim=(0, 1, 2))
    x = ((x - mean) / std).permute(0, 3, 1, 2).reshape(len(x), -1)  # channel-major, like a CHW tensor flattened
    y = nn.functional.one_hot(torch.as_tensor(labels[: c.examples]).long(), c.classes).float()
    return x, y


# %% [model] The network
def build(c: Config, generator):
    """3,072 → 200 → 200 → 10 with tanh between, PyTorch's default nn.Linear initialisation (weights and biases uniform
    in ±1/√fan_in), drawn from \`generator\` so a seed fixes the run."""
    sizes = [3 * 32 * 32, c.width, c.width, c.classes]
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        linear = nn.Linear(n_in, n_out)
        bound = 1 / math.sqrt(n_in)
        with torch.no_grad():
            linear.weight.copy_(torch.rand(n_out, n_in, generator=generator) * 2 * bound - bound)
            linear.bias.copy_(torch.rand(n_out, generator=generator) * 2 * bound - bound)
        layers.append(linear)
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# %% [loss] Squared loss
def loss_fn(model, x, y):
    """½ · ‖f(x) − y‖² averaged over the examples: the one-hot target's squared error."""
    return 0.5 * (model(x) - y).pow(2).sum(dim=1).mean()


# %% [gd-step] One step of full-batch gradient descent
def gd_step(model, x, y, lr):
    """θ ← θ − η · ∇L(θ), the gradient over all 5,000 examples at once: no minibatch noise, no momentum."""
    params = list(model.parameters())
    loss = loss_fn(model, x, y)
    grads = torch.autograd.grad(loss, params)
    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= lr * g
    return loss.detach(), torch.cat([g.reshape(-1) for g in grads])


# %% [hvp] Hessian–vector product
def hvp(model, x, y, v):
    """H·v without forming H (656,810 × 656,810): differentiate ⟨∇L, v⟩ once more (double backpropagation)."""
    params = list(model.parameters())
    grads = torch.autograd.grad(loss_fn(model, x, y), params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    hv = torch.autograd.grad(flat @ v, params)
    return torch.cat([h.reshape(-1) for h in hv])


# %% [lanczos] Sharpness by Lanczos
def lanczos(matvec, n, steps, start, k=3):
    """The k largest eigenvalues of a symmetric matrix seen only through matvec, and the top eigenvector. Lanczos builds
    an orthonormal basis of the Krylov space {v, Hv, H²v, …} (re-orthogonalized against every earlier vector) in which
    H is tridiagonal; that small matrix's top eigenvalues converge to H's first."""
    q = start / start.norm()
    basis, alphas, betas = [q], [], []
    for j in range(steps):
        w = matvec(basis[-1])
        alphas.append(float(w @ basis[-1]))
        for b in basis:  # full re-orthogonalization, twice is enough
            w = w - (w @ b) * b
        for b in basis:
            w = w - (w @ b) * b
        beta = float(w.norm())
        if j == steps - 1 or beta < 1e-10:
            break
        betas.append(beta)
        basis.append(w / beta)
    m = len(alphas)
    t = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if m > 1:
        off = torch.tensor(betas[: m - 1], dtype=torch.float64)
        t += torch.diag(off, 1) + torch.diag(off, -1)
    values, vectors = torch.linalg.eigh(t)
    top = torch.stack(basis[:m], dim=1) @ vectors[:, -1].to(start.dtype)
    return values.flip(0)[:k], top / top.norm()


def sharpness(model, x, y, c: Config, start=None, generator=None):
    """The Hessian's largest eigenvalues at the current weights, warm-started from the last top eigenvector."""
    n = sum(p.numel() for p in model.parameters())
    if start is None:
        start = torch.randn(n, generator=generator)
    return lanczos(lambda v: hvp(model, x, y, v), n, c.lanczos_steps, start)


# %% [stability] Why 2/η
def quadratic_iterates(curvature, lr, x0, steps):
    """Gradient descent on L(x) = ½·λ·x²: x ← x − η·λ·x = (1 − ηλ)·x. The distance to the minimum shrinks while
    |1 − ηλ| < 1, i.e. λ < 2/η; at λ > 2/η every step overshoots further and the iterate flips sign and grows."""
    xs = [x0]
    for _ in range(steps):
        xs.append(xs[-1] - lr * curvature * xs[-1])
    return torch.tensor(xs, dtype=torch.float64)
`;export{n as default};
