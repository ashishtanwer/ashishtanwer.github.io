const e=`# Lion, as the Pretraining Explainer trains its small GPT with it (4 layers, d_model 256, 4 heads, context 256,
# ≈16.1M parameters, on TinyStories): every weight moves by exactly ±lr, the sign of a blend of its momentum and its
# new gradient. One state tensor per weight tensor, half of AdamW's. scripts/pretraining/record_lion.py trains with
# this class; the check compares it with timm.optim.Lion.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    lr: float = 5e-4  # the peak learning rate, the best of the sweep; warmup and cosine decay scale it each step
    beta1: float = 0.9  # β₁: the blend that is signed leans on the momentum by β₁
    beta2: float = 0.99  # β₂: the momentum itself averages roughly the last 100 gradients
    weight_decay: float = 1.0  # λ, on the matrices and the embedding only; 10× AdamW's, as lr is 8× smaller


# %% [interpolate] The interpolated momentum
def interpolate(m, grad, c: Config):
    """c = β₁·m + (1 − β₁)·g: the momentum, nudged toward the new gradient."""
    return c.beta1 * m + (1 - c.beta1) * grad


# %% [lion-step] One weight tensor's step
@torch.no_grad()
def lion_step(param, grad, m, lr, weight_decay, c: Config):
    """θ ← θ − lr·λ·θ − lr · sign(c), then m ← β₂·m + (1 − β₂)·g. Returns the new momentum."""
    update = torch.sign(interpolate(m, grad, c))  # +1 or −1 for every weight (0 only where c is exactly 0)
    param *= 1 - lr * weight_decay
    param -= lr * update
    return c.beta2 * m + (1 - c.beta2) * grad  # a slower average than the one that was signed


# %% [param-groups] Which weights decay
def param_groups(named_params):
    """As AdamW: matrices and the (tied) embedding decay; LayerNorm gains and biases do not."""
    decay, no_decay = [], []
    for name, p in named_params:
        (decay if p.ndim == 2 else no_decay).append(p)
    return decay, no_decay


# %% [optimizer] Every weight of the model
class Lion:
    def __init__(self, named_params, c: Config):
        decay, no_decay = param_groups(named_params)
        self.groups = [(decay, c.weight_decay), (no_decay, 0.0)]
        self.c = c
        self.state = {p: torch.zeros_like(p) for p in decay + no_decay}  # m, one per weight tensor

    def zero_grad(self):
        for p in self.state:
            p.grad = None

    @torch.no_grad()
    def step(self, lr):
        for params, weight_decay in self.groups:
            for p in params:
                self.state[p] = lion_step(p, p.grad, self.state[p], lr, weight_decay, self.c)


# %% [train-step] One training step
def train_step(model, x, y, optimizer, lr):
    logits = model(x)  # (batch, context, vocab)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())  # mean next-token loss, nats
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(lr)
    return loss.detach()
`;export{e as default};
