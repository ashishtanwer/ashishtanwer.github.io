const e=`# Muon, as the Pretraining Explainer trains its small GPT with it (4 layers, d_model 256, 4 heads, context 256,
# ≈16.1M parameters, on TinyStories): for every hidden matrix, a momentum whose singular values Newton–Schulz
# iterations push toward 1, so the step changes every direction of the matrix by a similar amount. The embeddings and
# the LayerNorms are trained by AdamW with the AdamW tab's settings. scripts/pretraining/record_muon.py trains with
# this class; the check compares it with torch.optim.Muon and torch.optim.AdamW.
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# the AdamW tab's code, for the weights Muon does not take
_spec = importlib.util.spec_from_file_location('pretraining_adamw', Path(__file__).with_name('pretraining_adamw.py'))
adamw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adamw)


@dataclass
class Config:
    lr: float = 0.02  # the peak learning rate of the hidden matrices, the best of the sweep
    momentum: float = 0.95  # μ
    nesterov: bool = True
    ns_steps: int = 5
    ns_coefficients: tuple = (3.4445, -4.7750, 2.0315)  # a, b, c of the quintic below
    eps: float = 1e-7
    weight_decay: float = 0.0  # none on the hidden matrices, as in Keller Jordan's Muon


# %% [newton-schulz] Newton–Schulz orthogonalization
def newton_schulz(G, c: Config, iterates=None):
    """X ← a·X + (b·A + c·A²)·X with A = X·Xᵀ, five times, in bfloat16. Each step maps every singular value σ of X
    to aσ + bσ³ + cσ⁵ and keeps the singular vectors, so starting from G/‖G‖ (every σ ≤ 1) the σs rise toward 1:
    the result is close to U·Vᵀ, G = U·S·Vᵀ with its scales S thrown away."""
    a, b, c5 = c.ns_coefficients
    X = G.bfloat16()
    tall = G.shape[0] > G.shape[1]
    if tall:
        X = X.T  # iterate on the wide side, so A = X·Xᵀ is the smaller Gram matrix
    X = X / X.norm().clamp(min=c.eps)  # Frobenius norm 1, so no singular value exceeds 1
    if iterates is not None:
        iterates.append(X)
    for _ in range(c.ns_steps):
        A = X @ X.T  # the Gram matrix, (rows, rows)
        B = torch.addmm(A, A, A, beta=b, alpha=c5)  # b·A + c·A², rounded to bfloat16 once
        X = torch.addmm(X, B, X, beta=a)  # a·X + B·X
        if iterates is not None:
            iterates.append(X)
    return X.T if tall else X


# %% [muon-step] One hidden matrix's step
def muon_momentum(buf, grad, c: Config):
    """B ← μ·B + (1 − μ)·g. With Nesterov the matrix to orthogonalize looks one step ahead, (1 − μ)·g + μ·B."""
    buf = buf.lerp(grad, 1 - c.momentum)
    update = grad.lerp(buf, c.momentum) if c.nesterov else buf
    return buf, update


@torch.no_grad()
def muon_step(param, grad, buf, lr, c: Config, iterates=None):
    """θ ← θ − lr·√max(1, rows/cols)·O, O the orthogonalized update. Returns the new buffer."""
    buf, update = muon_momentum(buf, grad, c)
    O = newton_schulz(update, c, iterates)
    scale = max(1, param.shape[0] / param.shape[1]) ** 0.5  # a tall matrix (an MLP's 1024 × 256 up-projection) gets 2
    param *= 1 - lr * c.weight_decay
    param -= lr * scale * O.float()
    return buf


# %% [param-groups] Hidden matrices to Muon, the rest to AdamW
def param_groups(named_params):
    """Muon takes the matrices inside the blocks (attention q, k, v, o; the MLP's two). The token embedding (also the
    output head), the position embedding and the LayerNorms go to AdamW: an embedding row is a lookup, not a map
    whose directions should change evenly."""
    muon, other = [], []
    for name, p in named_params:
        (muon if p.ndim == 2 and name.startswith('blocks.') else other).append((name, p))
    return muon, other


# %% [optimizer] Every weight of the model
class Muon:
    def __init__(self, named_params, c: Config, adam: 'adamw.Config'):
        muon, other = param_groups(named_params)
        self.c, self.adam = c, adamw.AdamW(other, adam)
        self.state = {p: torch.zeros_like(p) for _, p in muon}  # the momentum buffer B, one per hidden matrix

    def zero_grad(self):
        self.adam.zero_grad()
        for p in self.state:
            p.grad = None

    @torch.no_grad()
    def step(self, lr, adam_lr):
        for p in self.state:
            self.state[p] = muon_step(p, p.grad, self.state[p], lr, self.c)
        self.adam.step(adam_lr)


# %% [train-step] One training step
def train_step(model, x, y, optimizer, lr, adam_lr):
    logits = model(x)  # (batch, context, vocab)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())  # mean next-token loss, nats
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(lr, adam_lr)
    return loss.detach()
`;export{e as default};
