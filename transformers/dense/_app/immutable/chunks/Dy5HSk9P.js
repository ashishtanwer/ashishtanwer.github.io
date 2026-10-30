const n=`# DeiT-tiny (facebook/deit-tiny-patch16-224), the model in the ViT tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    image_size: int = 224
    patch_size: int = 16  # 224 / 16 = a 14 × 14 grid of 196 patches
    channels: int = 3
    d_model: int = 192
    n_heads: int = 3  # 192 / 3 = 64 dimensions per head
    n_layers: int = 12
    d_mlp: int = 768  # 4 × d_model
    n_classes: int = 1000  # ImageNet-1k
    eps: float = 1e-12


# %% [patch-embedding] Patches → Wpatch
class PatchEmbedding(nn.Module):
    """Cut the image into 16 × 16 patches and project each one to d_model numbers."""

    def __init__(self, c: Config):
        super().__init__()
        # A 16 × 16 convolution with stride 16 is the same as flattening every patch to
        # 16 · 16 · 3 = 768 numbers and multiplying by Wpatch (768 → d_model).
        self.projection = nn.Conv2d(c.channels, c.d_model, kernel_size=c.patch_size, stride=c.patch_size)

    def forward(self, pixels):  # pixels: (batch, 3, 224, 224)
        grid = self.projection(pixels)  # (batch, d_model, 14, 14)
        return grid.flatten(2).transpose(1, 2)  # (batch, 196, d_model), patches in reading order


# %% [embedding] Patch embedding
class Embedding(nn.Module):
    """Patch vectors with a learned [CLS] vector in front, plus a learned position embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.patches = PatchEmbedding(c)
        n_patches = (c.image_size // c.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.d_model))
        self.position = nn.Parameter(torch.zeros(1, n_patches + 1, c.d_model))  # one row per [CLS] + patch

    def forward(self, pixels):
        patches = self.patches(pixels)  # (batch, 196, d_model)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        return torch.cat([cls, patches], dim=1) + self.position  # x₀: (batch, 197, d_model)


# %% [attention] Bidirectional self-attention
class Attention(nn.Module):
    """Every token attends to all 197 tokens; there is no causal mask in an image encoder."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        # Split into heads: (batch, heads, seq, head_dim)
        q, k, v = (proj(x).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        heads = weights @ v  # (batch, heads, seq, head_dim)
        return self.out(heads.transpose(1, 2).reshape(b, t, d))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))  # exact (erf) GELU


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c)
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final LayerNorm, then the classifier Wcls reads the [CLS] token only."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_f = nn.LayerNorm(c.d_model, eps=c.eps)
        self.classifier = nn.Linear(c.d_model, c.n_classes)

    def forward(self, x):  # x: (batch, 197, d_model)
        return self.classifier(self.ln_f(x)[:, 0])  # logits: (batch, n_classes)


def class_probabilities(logits):
    return logits.softmax(dim=-1)  # one probability per ImageNet class


# %% [model] Full model
class DeiT(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)

    def forward(self, pixels):  # pixels: (batch, 3, 224, 224), normalized
        x = self.embedding(pixels)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# probabilities = class_probabilities(DeiT()(pixels))
`;export{n as default};
