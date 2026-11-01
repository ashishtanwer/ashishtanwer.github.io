const e=`# A small convolutional network trained on MNIST, LeNet-style (LeCun et al. 1998): the Before Transformers Explainer's
# CNN tab as plain PyTorch. 6 learned 5 × 5 kernels slide over the digit, ReLU, 2 × 2 max pooling, 16 kernels over
# those maps, ReLU, pooling again, and one dense layer from the 16 × 5 × 5 pooled maps to the 10 digits. Trained like
# the MLP tab (same images, same batches, plain SGD at lr 0.1 for 3 epochs, seed 0), so the two compare directly.
# scripts/before-transformers/record_cnn.py trains the tab's network with exactly this code.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    channels: tuple = (1, 6, 16)  # LeNet-5's widths: 6 kernels on the digit, then 16 on their maps
    kernel: int = 5
    padding: tuple = (2, 0)  # the first convolution keeps 28 × 28, the second shrinks 14 → 10
    pool: int = 2
    classes: int = 10
    lr: float = 0.1
    batch_size: int = 64
    epochs: int = 3
    seed: int = 0


# %% [input] Pixels to an image
def to_input(images):
    """uint8 images (batch, 28, 28) → (batch, 1, 28, 28) in 0 … 1: one input channel, the picture kept 2-D."""
    return images[:, None].float() / 255


# %% [conv] A convolution, written out
def conv2d(x, weight, bias, padding=0):
    """x: (batch, c_in, H, W); weight: (c_out, c_in, k, k). Every kernel visits every position: the output pixel there
    is the k × k × c_in window under it dotted with the kernel, plus the kernel's bias. The same few weights serve
    every position (weight sharing)."""
    k = weight.shape[-1]
    x = F.pad(x, (padding,) * 4)  # zeros around the border, so a 5 × 5 kernel with padding 2 keeps the size
    windows = x.unfold(2, k, 1).unfold(3, k, 1)  # (batch, c_in, H_out, W_out, k, k): the window at each position
    return torch.einsum("bcyxij,ocij->boyx", windows, weight) + bias[:, None, None]


# %% [pool] Max pooling
def max_pool(x, size=2):
    """Each size × size window → its largest value, and which of its size² cells won (0 … size² − 1, row by row; on a
    tie, the first). Only the winner's gradient flows back."""
    b, c, h, w = x.shape
    windows = x.unfold(2, size, size).unfold(3, size, size).reshape(b, c, h // size, w // size, size * size)
    values, winner = windows.max(-1)
    return values, winner


# %% [model] The network
class CNN(nn.Module):
    """conv → ReLU → pool → conv → ReLU → pool → dense. nn.Conv2d computes conv2d above; the dense layer reads the
    16 pooled 5 × 5 maps laid end to end (400 values)."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        c0, c1, c2 = c.channels
        self.conv1 = nn.Conv2d(c0, c1, c.kernel, padding=c.padding[0])
        self.conv2 = nn.Conv2d(c1, c2, c.kernel, padding=c.padding[1])
        self.fc = nn.Linear(c2 * 5 * 5, c.classes)
        self.size = c.pool

    def forward(self, x):  # x: (batch, 1, 28, 28)
        z1 = self.conv1(x)  # (batch, 6, 28, 28)
        p1, win1 = max_pool(torch.relu(z1), self.size)  # (batch, 6, 14, 14)
        z2 = self.conv2(p1)  # (batch, 16, 10, 10)
        p2, win2 = max_pool(torch.relu(z2), self.size)  # (batch, 16, 5, 5)
        logits = self.fc(p2.flatten(1))  # (batch, 10)
        return {"x": x, "z1": z1, "p1": p1, "win1": win1, "z2": z2, "p2": p2, "win2": win2, "logits": logits}


# %% [receptive-field] The receptive field
def receptive_field(layers):
    """layers: (kernel, stride) from the input up. After each layer: how many input pixels across one unit sees, and
    the jump, how many input pixels apart two neighbouring units' windows start. A k-wide layer adds (k − 1) · jump."""
    size, jump, out = 1, 1, []
    for k, stride in layers:
        size += (k - 1) * jump
        jump *= stride
        out.append((size, jump))
    return out


# %% [weight-sharing] Weight sharing
@torch.no_grad()
def as_dense(conv, height, width):
    """The convolution as an ordinary dense layer, (outputs × inputs): the row of output pixel (o, y, x) holds kernel o
    at the input pixels its window covers and zeros everywhere else. Every row reuses the same few kernel weights."""
    c_out, c_in, k, _ = conv.weight.shape
    pad = conv.padding[0]
    h_out, w_out = height + 2 * pad - k + 1, width + 2 * pad - k + 1
    matrix = torch.zeros(c_out, h_out, w_out, c_in, height, width)
    for y in range(h_out):
        for x in range(w_out):
            for i in range(k):
                for j in range(k):
                    r, s = y + i - pad, x + j - pad  # the input pixel under kernel cell (i, j); outside = padding
                    if 0 <= r < height and 0 <= s < width:
                        matrix[:, y, x, :, r, s] = conv.weight[:, :, i, j]
    return matrix.view(c_out * h_out * w_out, c_in * height * width)


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


# %% [kernel-grad] A kernel's gradient
def kernel_grad(x, grad_out, k, padding=0):
    """∂L/∂W of a convolution from its input x (batch, c_in, H, W) and ∂L/∂(its output) (batch, c_out, H_out, W_out):
    at every position the kernel visited, the window there times the output's gradient at that position, summed over
    all positions and images. One kernel serves every position, so its gradient adds them all up."""
    x = F.pad(x, (padding,) * 4)
    windows = x.unfold(2, k, 1).unfold(3, k, 1)  # (batch, c_in, H_out, W_out, k, k)
    return torch.einsum("bcyxij,boyx->ocij", windows, grad_out), grad_out.sum((0, 2, 3))


# %% [sgd-step] One SGD step
@torch.no_grad()
def sgd_step(model, grads, lr):
    """w ← w − lr · ∂L/∂w for every kernel weight, bias and dense weight."""
    for p, g in zip(model.parameters(), grads):
        p -= lr * g


# %% [train] Training loop
def batches(n, c: Config, generator):
    """A new random order of the n training images every epoch, cut into full batches (the MLP tab's order)."""
    for _ in range(c.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n - c.batch_size + 1, c.batch_size):
            yield order[start : start + c.batch_size]


def train_step(model, images, labels, c: Config):
    """Forward, cross-entropy, autograd's backward pass, the SGD step; returns what the step saw before the update."""
    acts = model(to_input(images))
    loss = F.cross_entropy(acts["logits"], labels)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    sgd_step(model, grads, c.lr)
    return loss.detach(), acts, grads


def train(images, labels, c: Config = Config()):
    torch.manual_seed(c.seed)  # initial weights: PyTorch's defaults, uniform in ±1/√(inputs per unit)
    model = CNN(c)
    generator = torch.Generator().manual_seed(c.seed)
    for idx in batches(len(labels), c, generator):
        train_step(model, images[idx], labels[idx], c)
    return model
`;export{e as default};
