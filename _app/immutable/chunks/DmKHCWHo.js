const n=`# Grokking on modular addition, the grokking tab, as plain PyTorch.
# A one-layer transformer reads "a b =" and predicts (a + b) mod 113. Trained full-batch on 30 % of the 12,769 pairs
# with AdamW and weight decay 1, it first memorizes the training pairs (train accuracy 100 %, test at chance) and only
# thousands of steps later generalizes. In between, it builds a circuit out of a few Fourier frequencies: it embeds
# each number as cos(wa), sin(wa) for a handful of w, multiplies them in the MLP into cos(w(a + b)), sin(w(a + b)) and
# reads out c through cos(w(a + b − c)). Weight decay then removes the memorizing weights. Setup, initialization and
# the progress measures follow Nanda et al. 2023 (arXiv:2301.05217, code: mechanistic-interpretability-grokking/
# progress-measures-paper @ 23d2c64); the task is Power et al. 2022's (arXiv:2201.02177).
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    p: int = 113  # the modulus; tokens 0 … 112 are numbers, token 113 is "="
    d_model: int = 128
    heads: int = 4
    d_head: int = 32
    d_mlp: int = 512
    frac_train: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple = (0.9, 0.98)
    warmup: int = 10  # steps of linear learning-rate warmup
    data_seed: int = 0
    model_seed: int = 0


# %% [data] Every pair (a, b), 30 % of them to train on
def dataset(c: Config = Config()):
    """All p² inputs "a b =" in row-major order (a, b), their labels (a + b) mod p, and which ones are training pairs:
    the paper's split, Python's random.shuffle of the pairs with seed data_seed, the first frac_train of them."""
    pairs = [(a, b) for a in range(c.p) for b in range(c.p)]
    shuffled = pairs.copy()
    random.Random(c.data_seed).shuffle(shuffled)
    train = set(shuffled[: int(c.frac_train * len(pairs))])
    x = torch.tensor([(a, b, c.p) for a, b in pairs])
    is_train = torch.tensor([pair in train for pair in pairs])
    return x, (x[:, 0] + x[:, 1]) % c.p, is_train


# %% [model] One attention layer and one MLP, no LayerNorm
def init_params(c: Config = Config()):
    """The paper's initialization: Gaussian weights scaled by 1/√fan-in (the unembedding by 1/√vocab), zero biases."""
    g = torch.Generator().manual_seed(c.model_seed)
    d, H, e = c.d_model, c.heads, c.d_head
    shapes = {"W_E": ((d, c.p + 1), d), "W_pos": ((3, d), d), "W_K": ((H, e, d), d), "W_Q": ((H, e, d), d),
              "W_V": ((H, e, d), d), "W_O": ((d, H * e), d), "W_in": ((c.d_mlp, d), d), "W_out": ((d, c.d_mlp), d),
              "W_U": ((d, c.p), c.p)}
    params = {name: torch.randn(shape, generator=g) / fan ** 0.5 for name, (shape, fan) in shapes.items()}
    params["b_in"], params["b_out"] = torch.zeros(c.d_mlp), torch.zeros(d)
    return params


def forward(params, x, c: Config = Config()):
    """x: (B, 3) tokens "a b =" → logits over c at "=" (B, p), the attention "=" pays to a, b, = per head (B, H, 3) and
    the MLP's neurons at "=" (B, d_mlp). With one layer only the "=" position reaches the output, so only its query,
    MLP and unembedding are computed; a and b enter through the keys and values."""
    P, B = params, x.shape[0]
    h = P["W_E"].T[x] + P["W_pos"]  # (B, 3, d) residual stream
    q = torch.einsum("hed,bd->bhe", P["W_Q"], h[:, 2])
    k = torch.einsum("hed,bpd->bhpe", P["W_K"], h)
    v = torch.einsum("hed,bpd->bhpe", P["W_V"], h)
    attn = (torch.einsum("bhe,bhpe->bhp", q, k) / c.d_head ** 0.5).softmax(-1)
    z = torch.einsum("bhp,bhpe->bhe", attn, v).reshape(B, -1)
    r = h[:, 2] + z @ P["W_O"].T
    neurons = F.relu(r @ P["W_in"].T + P["b_in"])
    r = r + neurons @ P["W_out"].T + P["b_out"]
    return r @ P["W_U"], attn, neurons


def loss_fn(logits, labels):
    """Cross-entropy in float64: in float32, log-softmax of a very confident prediction can only return multiples of
    1.2e-7, which makes the late loss spike (the paper's note)."""
    return F.cross_entropy(logits.double(), labels)


# %% [train] Full-batch AdamW with weight decay
def optimizer(params, c: Config = Config()):
    """AdamW over the parameters in a fixed order, with the paper's 10-step linear warmup (the first step has lr 0)."""
    opt = torch.optim.AdamW(list(params.values()), lr=c.lr, weight_decay=c.weight_decay, betas=c.betas)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / c.warmup, 1))


def train_step(params, opt, sched, x, labels, c: Config = Config()):
    """One step on the whole training set; returns the loss and accuracy before the update."""
    logits = forward(params, x, c)[0]
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    sched.step()
    opt.zero_grad()
    return loss.item(), (logits.argmax(-1) == labels).double().mean().item()


def sum_sq_weights(params):
    """Σ w² over every parameter: what weight decay pulls down."""
    return sum(w.detach().double().pow(2).sum() for w in params.values())


# %% [fourier] The embedding in the Fourier basis
def fourier_basis(p: int = 113):
    """(p, p): the constant, then cos(2πkx/p) and sin(2πkx/p) for k = 1 … (p − 1)/2, each row of unit norm."""
    x = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64)]
    for k in range(1, p // 2 + 1):
        rows += [torch.cos(2 * torch.pi * k * x / p), torch.sin(2 * torch.pi * k * x / p)]
    basis = torch.stack(rows)
    return basis / basis.norm(dim=1, keepdim=True)


def embedding_spectrum(params, c: Config = Config()):
    """Norm of the number embeddings W_E[:, :p] along each basis row → (p,), and per frequency k the norm of its cos
    and sin rows together → (p // 2,). A generalizing model puts nearly all of it on a few key frequencies."""
    coeffs = fourier_basis(c.p) @ params["W_E"][:, : c.p].T.double()  # (p, d)
    norms = coeffs.norm(dim=1)
    return norms, (norms[1::2] ** 2 + norms[2::2] ** 2).sqrt()


def gini(values):
    """Gini coefficient of non-negative values: 0 when all are equal, near 1 when one holds everything."""
    v = values.double().sort().values
    n = v.numel()
    return ((2 * torch.arange(1, n + 1, dtype=torch.float64) - n - 1) * v).sum() / (n * v.sum())


# %% [neurons] Each neuron as a function of (a, b), and the key frequencies
def fft2d(grid, p: int = 113):
    """(p², …) values over (a, b) → the same in the 2D Fourier basis (rows: a's basis term, columns: b's)."""
    basis = fourier_basis(p)
    g = grid.double().reshape(p, p, -1)
    return torch.einsum("abn,fa,Fb->fFn", g, basis, basis)


def neuron_frequencies(neurons, p: int = 113):
    """neurons (p², n) at "=" → for every neuron and frequency k the fraction of its variance (mean removed) in the 3 × 3
    block of 2D Fourier terms {1, cos kx, sin kx} ⊗ {1, cos ky, sin ky}: (n, p // 2). A neuron that never fires has no
    variance; its shares are 0."""
    spec = fft2d(neurons - neurons.mean(0), p).pow(2)  # (p, p, n)
    total = spec.sum((0, 1)).clamp_min(torch.finfo(torch.float64).tiny)
    fracs = []
    for k in range(1, p // 2 + 1):
        idx = torch.tensor([0, 2 * k - 1, 2 * k])
        fracs.append(spec[idx][:, idx].sum((0, 1)) / total)
    return torch.stack(fracs, 1)


# %% [progress] Restricted and excluded loss
def sum_directions(p: int, freqs):
    """Unit vectors over the p² pairs along cos(w(a + b)) and sin(w(a + b)) for each key frequency, w = 2πk/p:
    (2·len(freqs), p²). cos(w(a+b)) = cos wa cos wb − sin wa sin wb, so these are 2D Fourier terms."""
    basis = fourier_basis(p)
    rows = []
    for k in freqs:
        ck, sk = basis[2 * k - 1], basis[2 * k]
        rows += [(ck[:, None] * ck[None] - sk[:, None] * sk[None]).flatten() / 2 ** 0.5,
                 (sk[:, None] * ck[None] + ck[:, None] * sk[None]).flatten() / 2 ** 0.5]
    return torch.stack(rows)


def restricted_loss(logits, labels, freqs, p: int = 113):
    """Loss on all p² pairs when, for every answer c, the logits keep only their cos(w(a + b)) and sin(w(a + b)) parts
    at the key frequencies, plus each answer's mean logit (the paper's bias correction). Falls once the circuit can
    carry the answer on its own."""
    u = sum_directions(p, freqs)
    L = logits.double()
    kept = u.T @ (u @ L)
    kept = kept + (L - kept).mean(0, keepdim=True)
    return F.cross_entropy(kept, labels)


def excluded_loss(logits, labels, is_train, freqs, p: int = 113):
    """Training loss with those same parts removed from the logits: it rises while the model starts relying on the
    circuit, even though the training loss itself is already near zero."""
    u = sum_directions(p, freqs)
    L = logits.double()
    rest = L - u.T @ (u @ L)
    return F.cross_entropy(rest[is_train], labels[is_train])
`;export{n as default};
