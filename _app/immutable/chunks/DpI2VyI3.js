const e=`# A multilayer perceptron trained on MNIST: the Before Transformers Explainer's MLP tab as plain PyTorch. 784 pixels
# → 64 → 32 → 10 digits with ReLU between, softmax and cross-entropy, backpropagation written out layer by layer,
# and plain minibatch SGD. scripts/before-transformers/record_mlp.py trains the tab's network with exactly this code.
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    sizes: tuple = (784, 64, 32, 10)  # 28 × 28 pixels in, one logit per digit out
    lr: float = 0.1
    batch_size: int = 64
    epochs: int = 3
    seed: int = 0


# %% [input] Pixels to a vector
def to_input(images):
    """uint8 images (batch, 28, 28) → (batch, 784): the 28 rows of pixels laid end to end, scaled to 0 … 1."""
    return images.flatten(1).float() / 255


# %% [forward] Forward pass
class MLP(nn.Module):
    """Three linear layers with ReLU after the first two. nn.Linear stores W as (out, in) and computes x Wᵀ + b:
    unit j of a layer is row j of W dotted with the layer's input, plus b_j."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        d_in, d1, d2, d_out = c.sizes
        self.fc1 = nn.Linear(d_in, d1)
        self.fc2 = nn.Linear(d1, d2)
        self.fc3 = nn.Linear(d2, d_out)

    def forward(self, x):  # x: (batch, 784)
        z1 = self.fc1(x)  # (batch, 64)
        h1 = torch.relu(z1)
        z2 = self.fc2(h1)  # (batch, 32)
        h2 = torch.relu(z2)
        logits = self.fc3(h2)  # (batch, 10)
        return {"x": x, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "logits": logits}


# %% [loss] Softmax and cross-entropy
def softmax(logits):
    e = (logits - logits.max(-1, keepdim=True).values).exp()  # shifting by the max changes nothing but overflow
    return e / e.sum(-1, keepdim=True)


def cross_entropy(logits, labels):
    """L = −log p(correct digit), averaged over the batch; log-softmax computed as z − log Σ exp z."""
    shifted = logits - logits.max(-1, keepdim=True).values
    log_p = shifted - shifted.exp().sum(-1, keepdim=True).log()
    return -log_p[torch.arange(len(labels)), labels].mean()


# %% [backward] Backpropagation by hand
@torch.no_grad()
def backward(model, acts, labels):
    """The chain rule from the loss back to every weight, one layer at a time. Each ∂L/∂(input of a layer) is the
    layer's output gradient times its W; ReLU lets a gradient through only where its input was positive; each
    ∂L/∂W is the output gradient times the layer's input. Returns the activations' gradients and the parameters'
    (in model.parameters() order)."""
    n = len(labels)
    g3 = softmax(acts["logits"])  # softmax and cross-entropy together: ∂L/∂logits = (p − onehot(y)) / batch
    g3[torch.arange(n), labels] -= 1
    g3 /= n
    dW3, db3 = g3.T @ acts["h2"], g3.sum(0)
    dh2 = g3 @ model.fc3.weight  # (batch, 32)
    g2 = dh2 * (acts["z2"] > 0)
    dW2, db2 = g2.T @ acts["h1"], g2.sum(0)
    dh1 = g2 @ model.fc2.weight  # (batch, 64)
    g1 = dh1 * (acts["z1"] > 0)
    dW1, db1 = g1.T @ acts["x"], g1.sum(0)  # (64, 784): row u is a weighted sum of the batch's images
    grads = {"logits": g3, "h2": dh2, "z2": g2, "h1": dh1, "z1": g1}
    return grads, [dW1, db1, dW2, db2, dW3, db3]


# %% [sgd-step] One SGD step
@torch.no_grad()
def sgd_step(model, param_grads, lr):
    """The whole optimizer: w ← w − lr · ∂L/∂w for every weight and bias."""
    for p, g in zip(model.parameters(), param_grads):
        p -= lr * g


# %% [train] Training loop
def batches(n, c: Config, generator):
    """A new random order of the n training images every epoch, cut into full batches (the last partial one is
    dropped)."""
    for _ in range(c.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n - c.batch_size + 1, c.batch_size):
            yield order[start : start + c.batch_size]


@torch.no_grad()  # no autograd anywhere: the backward pass is the hand-written one
def train_step(model, images, labels, c: Config):
    """Forward, loss, backward, update; returns what the step saw before the update."""
    acts = model(to_input(images))
    loss = cross_entropy(acts["logits"], labels)
    grads, param_grads = backward(model, acts, labels)
    sgd_step(model, param_grads, c.lr)
    return loss, acts, grads, param_grads


def train(images, labels, c: Config = Config()):
    torch.manual_seed(c.seed)  # initial weights: nn.Linear's default, uniform in ±1/√(inputs)
    model = MLP(c)
    generator = torch.Generator().manual_seed(c.seed)
    for idx in batches(len(labels), c, generator):
        train_step(model, images[idx], labels[idx], c)
    return model
`;export{e as default};
