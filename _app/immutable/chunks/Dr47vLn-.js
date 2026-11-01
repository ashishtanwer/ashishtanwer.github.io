const n=`# A generative adversarial network (Goodfellow et al. 2014) in the DCGAN style (Radford et al. 2016), trained locally on
# MNIST: the Before Transformers Explainer's GAN tab as plain PyTorch. A generator turns 64 random numbers into a
# 28 × 28 digit with transposed convolutions (1 × 1 → 7 × 7 → 14 × 14 → 28 × 28); a discriminator, an ordinary
# convolutional network, scores an image as real (a training digit) or fake (the generator's). They train in turns: the
# discriminator learns to tell them apart, then the generator learns to make the discriminator call its images real.
# Nobody shows the generator a digit; everything it learns comes through the discriminator's gradient.
# scripts/before-transformers/record_gan.py trains the tab's networks with exactly this code.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    z_dim: int = 64  # the noise vector the generator starts from
    g_channels: tuple = (128, 64)  # generator maps at 7 × 7 and 14 × 14
    d_channels: tuple = (64, 128)  # discriminator maps at 14 × 14 and 7 × 7
    lr: float = 2e-4  # DCGAN's Adam settings for both networks
    betas: tuple = (0.5, 0.999)
    batch_size: int = 64
    epochs: int = 15
    seed: int = 0


# %% [noise] The noise vector
def to_input(images):
    """uint8 MNIST images (batch, 28, 28) → (batch, 1, 28, 28) in −1 … 1, the range of the generator's tanh."""
    return images[:, None].float() / 127.5 - 1


def noise(n, c: Config, generator=None):
    """n vectors of z_dim independent standard normal numbers, shaped as 1 × 1 maps with z_dim channels."""
    return torch.randn(n, c.z_dim, 1, 1, generator=generator)


# %% [conv-transpose] A transposed convolution, written out
def conv_transpose2d(x, weight, stride=1, padding=0):
    """x: (batch, c_in, H, W); weight: (c_in, c_out, k, k). The mirror of a convolution: each input pixel stamps a copy
    of the kernel, scaled by its value, onto the output, \`stride\` pixels from its neighbour's stamp; overlapping stamps
    add up. Output size (H − 1) · stride − 2 · padding + k."""
    b, c_in, h, w = x.shape
    k = weight.shape[-1]
    full = torch.zeros(b, weight.shape[1], (h - 1) * stride + k, (w - 1) * stride + k)
    for y in range(h):
        for xx in range(w):
            stamp = torch.einsum("bc,cokl->bokl", x[:, :, y, xx], weight)  # (batch, c_out, k, k)
            full[:, :, y * stride : y * stride + k, xx * stride : xx * stride + k] += stamp
    return full[:, :, padding : full.shape[2] - padding, padding : full.shape[3] - padding]


# %% [generator] The generator
class Generator(nn.Module):
    """z (z_dim × 1 × 1) → 7 × 7 → 14 × 14 → 28 × 28. Batch norm keeps each layer's maps at a steady scale; in training
    it uses the batch's own statistics, when sampling the running averages it kept."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        c1, c2 = c.g_channels
        self.up1 = nn.ConvTranspose2d(c.z_dim, c1, 7, 1, 0, bias=False)  # 1 × 1 → 7 × 7
        self.bn1 = nn.BatchNorm2d(c1)
        self.up2 = nn.ConvTranspose2d(c1, c2, 4, 2, 1, bias=False)  # 7 × 7 → 14 × 14
        self.bn2 = nn.BatchNorm2d(c2)
        self.up3 = nn.ConvTranspose2d(c2, 1, 4, 2, 1)  # 14 × 14 → 28 × 28

    def forward(self, z):
        u1 = self.up1(z)
        a1 = torch.relu(self.bn1(u1))
        u2 = self.up2(a1)
        a2 = torch.relu(self.bn2(u2))
        u3 = self.up3(a2)
        return {"z": z, "u1": u1, "a1": a1, "u2": u2, "a2": a2, "u3": u3, "image": torch.tanh(u3)}


# %% [discriminator] The discriminator
class Discriminator(nn.Module):
    """28 × 28 → 14 × 14 → 7 × 7 → one number, the logit of 'real': strided convolutions with leaky ReLU (a small
    slope below zero keeps a gradient flowing back to the generator everywhere)."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        d1, d2 = c.d_channels
        self.conv1 = nn.Conv2d(1, d1, 4, 2, 1)  # 28 → 14
        self.conv2 = nn.Conv2d(d1, d2, 4, 2, 1)  # 14 → 7
        self.conv3 = nn.Conv2d(d2, 1, 7)  # 7 × 7 → 1 × 1

    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x), 0.2)
        h2 = F.leaky_relu(self.conv2(h1), 0.2)
        return {"h1": h1, "h2": h2, "logit": self.conv3(h2).flatten()}  # σ(logit) = D(x), the chance it is real


# %% [losses] The two losses
def d_loss(real_logit, fake_logit):
    """The discriminator wants D(x) → 1 on real digits and D(G(z)) → 0 on fakes:
    −mean log D(x) − mean log(1 − D(G(z))), with log σ(l) = −softplus(−l) for stability."""
    return F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()


def g_loss(fake_logit):
    """The generator wants the discriminator to call its images real: −mean log D(G(z)) (the 'non-saturating' loss,
    which still has a gradient when D confidently rejects the fakes, unlike minimising log(1 − D(G(z))))."""
    return F.softplus(-fake_logit).mean()


# %% [train-step] One alternating update
def train_step(G, D, opt_g, opt_d, real, z):
    """First the discriminator: real digits against the generator's current images (detached, so G is not changed).
    Then the generator: the same z through the updated discriminator, gradients through D into G; only G steps."""
    fake = G(z)["image"]
    real_logit, fake_logit = D(real)["logit"], D(fake.detach())["logit"]
    loss_d = d_loss(real_logit, fake_logit)
    opt_d.zero_grad()
    loss_d.backward()
    opt_d.step()

    fake_logit_after = D(fake)["logit"]  # the updated discriminator on the same fakes, now with G's graph attached
    loss_g = g_loss(fake_logit_after)
    opt_g.zero_grad()
    loss_g.backward()  # also fills D's gradients; opt_d.zero_grad() clears them before D's next step
    opt_g.step()
    return {"loss_d": loss_d.detach(), "loss_g": loss_g.detach(), "real_logit": real_logit.detach(),
            "fake_logit": fake_logit.detach(), "fake_logit_after": fake_logit_after.detach(), "fake": fake.detach()}


# %% [interpolate] Walking between two noise vectors
def interpolate(z_a, z_b, steps):
    """Points on the straight line from z_a to z_b; the generator maps each one to an image."""
    t = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
    return (1 - t) * z_a + t * z_b


# %% [train] Training loop
def batches(n, c: Config, generator):
    """A new random order of the n training images every epoch, cut into full batches."""
    for _ in range(c.epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n - c.batch_size + 1, c.batch_size):
            yield order[start : start + c.batch_size]


def init_weights(module):
    """DCGAN's initialisation: convolution weights N(0, 0.02²), batch-norm scales N(1, 0.02²) and shifts 0."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def build(c: Config = Config()):
    torch.manual_seed(c.seed)
    G, D = Generator(c), Discriminator(c)
    G.apply(init_weights)
    D.apply(init_weights)
    opt_g = torch.optim.Adam(G.parameters(), lr=c.lr, betas=c.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=c.lr, betas=c.betas)
    return G, D, opt_g, opt_d


def train(images, c: Config = Config()):
    G, D, opt_g, opt_d = build(c)
    generator = torch.Generator().manual_seed(c.seed)  # batch order and noise both come from it
    for idx in batches(len(images), c, generator):
        train_step(G, D, opt_g, opt_d, to_input(images[idx]), noise(c.batch_size, c, generator))
    return G, D
`;export{n as default};
