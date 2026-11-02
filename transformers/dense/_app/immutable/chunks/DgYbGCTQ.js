const e=`# SGD with momentum, as the Pretraining Explainer trains its small GPT with it (4 layers, d_model 256, 4 heads,
# context 256, ≈16.1M parameters, on TinyStories): one momentum buffer per weight tensor, and a step along it.
# scripts/pretraining/record_sgd.py trains with this class; the check compares it with torch.optim.SGD.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    lr: float = 0.25  # the peak learning rate, the best of the sweep; warmup and cosine decay scale it each step
    momentum: float = 0.9  # μ: how much of the last buffer every step keeps
    # no weight decay: every weight is moved by its gradient alone


# %% [momentum] The momentum buffer
def momentum_buffer(buf, grad, momentum):
    """b₁ = g₁, then b_t = μ·b_{t−1} + g_t: a sum of past gradients, each damped by μ for every step since."""
    if buf is None:
        return grad.clone()
    return momentum * buf + grad


# %% [sgd-step] One weight tensor's step
@torch.no_grad()
def sgd_step(param, grad, buf, lr, c: Config):
    """θ ← θ − lr · b. Returns the new buffer."""
    buf = momentum_buffer(buf, grad, c.momentum)
    param -= lr * buf
    return buf


# %% [optimizer] Every weight of the model
class SGD:
    def __init__(self, params, c: Config):
        self.params = list(params)
        self.c = c
        self.state = {p: None for p in self.params}  # one buffer per weight tensor, created at the first step

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, lr):
        for p in self.params:
            self.state[p] = sgd_step(p, p.grad, self.state[p], lr, self.c)


# %% [train-step] One training step
def train_step(model, x, y, optimizer, lr):
    logits = model(x)  # (batch, context, vocab)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())  # mean next-token loss, nats
    optimizer.zero_grad()
    loss.backward()
    optimizer.step(lr)
    return loss.detach()
`;export{e as default};
