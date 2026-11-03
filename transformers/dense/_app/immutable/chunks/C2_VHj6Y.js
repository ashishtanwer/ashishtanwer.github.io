const e=`# Chinchilla scaling (Hoffmann et al. 2022), fitted to the Pretraining Explainer's own grid of small GPTs (4 layers,
# 4 heads, d_model 32 … 256, AdamW on TinyStories, 2 … 16M tokens): the final loss of a model with N parameters
# trained on D tokens is modelled as L(N, D) = E + A / N^α + B / D^β, fitted the paper's third way (a Huber loss on
# log L, minimised by L-BFGS from a grid of starting points), and the fit gives the compute-optimal split of a budget
# C ≈ 6·N·D between parameters and tokens. scripts/pretraining/record_chinchilla.py fits the tab's grid with these
# functions. At this scale (a few million parameters, minutes of training) the numbers are a toy, not Chinchilla's.
import itertools
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    huber_delta: float = 1e-3  # the paper's δ
    alphas: tuple = (0.25, 0.5, 1.0)  # the starting grid for α and β (the paper's is finer: 0 … 2 in steps of 0.5)
    log_a: tuple = (0.0, 5.0, 10.0, 15.0, 20.0)  # … for log A and log B (0 … 25 in steps of 5)
    log_e: tuple = (-1.0, 0.0, 1.0)  # … for log E (−1 … 1 in steps of 0.5)
    lbfgs_steps: int = 200


# %% [law] The loss law
def loss_law(n, d, params):
    """E: the loss of an infinitely large model trained for ever (the entropy of the text, and what no model this
    shape can fix); A/N^α: what a model of N parameters cannot fit; B/D^β: what D tokens cannot teach."""
    return params["E"] + params["A"] / n ** params["alpha"] + params["B"] / d ** params["beta"]


# %% [fit] Fitting the law
def log_law(theta, log_n, log_d):
    """log L̂ = LSE(a − α·log N, b − β·log D, e), with A = eᵃ, B = eᵇ, E = eᵉ: the law in log space, so the three terms
    stay positive and a plain optimiser can move a, b and e freely."""
    a, b, e, alpha, beta = theta
    return torch.logsumexp(torch.stack([a - alpha * log_n, b - beta * log_d, e.expand_as(log_n)]), dim=0)


def huber(r, delta):
    return torch.where(r.abs() <= delta, 0.5 * r**2, delta * (r.abs() - 0.5 * delta))


def objective(theta, log_n, log_d, log_l, c: Config):
    """The sum of Huber losses between log L̂ and the measured log L: near-squared for small misfits, linear for large
    ones, so one bad run cannot drag the fit."""
    return huber(log_law(theta, log_n, log_d) - log_l, c.huber_delta).sum()


def fit(n, d, loss, c: Config):
    """L-BFGS from every point of the starting grid; the best end point wins."""
    log_n, log_d, log_l = (torch.log(torch.as_tensor(v, dtype=torch.float64)) for v in (n, d, loss))
    best = None
    for start in itertools.product(c.log_a, c.log_a, c.log_e, c.alphas, c.alphas):
        theta = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([theta], max_iter=c.lbfgs_steps, tolerance_grad=1e-12, tolerance_change=1e-15,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            value = objective(theta, log_n, log_d, log_l, c)
            value.backward()
            return value

        optimizer.step(closure)
        value = float(objective(theta.detach(), log_n, log_d, log_l, c))
        if math.isfinite(value) and (best is None or value < best[0]):
            best = (value, theta.detach().clone(), start)
    a, b, e, alpha, beta = best[1].tolist()
    return {"A": math.exp(a), "B": math.exp(b), "E": math.exp(e), "alpha": alpha, "beta": beta,
            "objective": best[0], "start": list(best[2])}


# %% [compute] Compute
def flops(n, d):
    """C ≈ 6·N·D: a forward pass costs about 2 FLOPs (one multiply-add) per parameter per token, the backward pass
    twice that."""
    return 6 * n * d


# %% [frontier] The compute-optimal frontier
def optimal_split(compute, params):
    """Minimising L(N, C / 6N) over N gives N_opt = G·(C/6)^a and D_opt = G⁻¹·(C/6)^b with a = β/(α+β), b = α/(α+β)
    and G = (αA / βB)^(1/(α+β)): the paper's closed form for its third approach."""
    alpha, beta = params["alpha"], params["beta"]
    g = (alpha * params["A"] / (beta * params["B"])) ** (1 / (alpha + beta))
    n_opt = g * (compute / 6) ** (beta / (alpha + beta))
    return n_opt, (compute / 6) / n_opt
`;export{e as default};
