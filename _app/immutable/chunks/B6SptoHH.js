const e=`# AdamW, as the Pretraining Explainer trains its small GPT with it (4 layers, d_model 256, 4 heads, context 256,
# ≈16.1M parameters, on TinyStories): a running mean m and mean square v of every weight's gradient, both corrected
# for starting at zero, and a weight decay applied apart from them. scripts/pretraining/record_adamw.py trains with
# this class; the check compares it with torch.optim.AdamW.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    lr: float = 4e-3  # the peak learning rate, the best of the sweep; warmup and cosine decay scale it each step
    beta1: float = 0.9  # β₁: the first moment averages roughly the last 1 / (1 − β₁) = 10 gradients
    beta2: float = 0.95  # β₂: the second moment roughly the last 20 squared gradients
    eps: float = 1e-8
    weight_decay: float = 0.1  # λ, on the matrices and the embedding only


# %% [moments] First and second moments
def moments(m, v, grad, c: Config):
    m = c.beta1 * m + (1 - c.beta1) * grad  # a running mean of the gradient
    v = c.beta2 * v + (1 - c.beta2) * grad * grad  # and of its square, per weight
    return m, v


# %% [bias-correction] Bias correction
def bias_correction(m, v, t, c: Config):
    """m and v start at zero, so after t steps they are too small by a factor 1 − βᵗ; dividing undoes it."""
    return m / (1 - c.beta1**t), v / (1 - c.beta2**t)


# %% [adamw-step] One weight tensor's step, with decoupled weight decay
@torch.no_grad()
def adamw_step(param, grad, state, lr, weight_decay, c: Config):
    """θ ← θ − lr·λ·θ − lr · m̂ / (√v̂ + ε). The decay shrinks θ directly instead of joining the gradient,
    so m and v never see it."""
    state['t'] += 1
    param *= 1 - lr * weight_decay
    state['m'], state['v'] = moments(state['m'], state['v'], grad, c)
    m_hat, v_hat = bias_correction(state['m'], state['v'], state['t'], c)
    param -= lr * m_hat / (v_hat.sqrt() + c.eps)


# %% [param-groups] Which weights decay
def param_groups(named_params):
    """Matrices and the (tied) embedding decay; LayerNorm gains and biases, vectors, do not."""
    decay, no_decay = [], []
    for name, p in named_params:
        (decay if p.ndim == 2 else no_decay).append(p)
    return decay, no_decay


# %% [optimizer] Every weight of the model
class AdamW:
    def __init__(self, named_params, c: Config):
        decay, no_decay = param_groups(named_params)
        self.groups = [(decay, c.weight_decay), (no_decay, 0.0)]
        self.c = c
        self.state = {p: {'t': 0, 'm': torch.zeros_like(p), 'v': torch.zeros_like(p)} for p in decay + no_decay}

    def zero_grad(self):
        for p in self.state:
            p.grad = None

    @torch.no_grad()
    def step(self, lr):
        for params, weight_decay in self.groups:
            for p in params:
                adamw_step(p, p.grad, self.state[p], lr, weight_decay, self.c)


# %% [train-step] One training step
def train_step(model, x, y, optimizer, lr):
    logits = model(x)  # (batch, context, vocab)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())  # mean next-token loss, nats
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(lr)
    return loss.detach()
`;export{e as default};
