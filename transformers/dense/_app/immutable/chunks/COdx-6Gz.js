const e=`# ResNet-50 (He et al., 2015) as released in microsoft/resnet-50 (transformers' ResNetForImageClassification): the
# Before Transformers Explainer's ResNet tab as plain PyTorch. A 7 × 7 stem, four stages of bottleneck blocks
# (3, 4, 6, 3) that halve the grid and widen the channels, global average pooling and one linear layer over the 1,000
# ImageNet classes. scripts/before-transformers/record_resnet.py runs the tab's photo through exactly this code.
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass
class Config:
    image_size: int = 224
    resize: int = 256  # int(224 / crop_pct 0.875): shortest side to 256, then the centre 224 × 224
    mean: tuple = (0.485, 0.456, 0.406)  # ImageNet's per-channel mean and standard deviation
    std: tuple = (0.229, 0.224, 0.225)
    stem_channels: int = 64
    widths: tuple = (256, 512, 1024, 2048)  # output channels of the four stages
    depths: tuple = (3, 4, 6, 3)  # bottleneck blocks per stage: 16 in all, 3 convolutions each
    reduction: int = 4  # a block's inner width is its output width / 4
    num_classes: int = 1000


# %% [preprocess] The image the network sees
def preprocess(image, c: Config = Config()):
    """PIL image → (1, 3, 224, 224): shortest side to 256 (bicubic), the centre 224 × 224, pixels to 0 … 1, then each
    colour channel shifted and scaled by ImageNet's mean and standard deviation."""
    image = image.convert("RGB")
    w, h = image.size
    if w <= h:
        size = (c.resize, int(c.resize * h / w))  # PIL sizes are (width, height)
    else:
        size = (int(c.resize * w / h), c.resize)
    image = image.resize(size, Image.BICUBIC)
    left, top = (image.width - c.image_size) // 2, (image.height - c.image_size) // 2
    image = image.crop((left, top, left + c.image_size, top + c.image_size))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
    mean, std = torch.tensor(c.mean)[:, None, None], torch.tensor(c.std)[:, None, None]
    return ((x - mean) / std)[None]


# %% [conv-bn] Convolution, batch norm, ReLU
class ConvBN(nn.Module):
    """A k × k convolution without bias; then batch norm with the statistics saved in training,
    y = (conv(x) − running_mean) / √(running_var + ε) · γ + β per channel; then ReLU unless the layer ends a block.
    Each output pixel of each output channel sums k × k pixels of every input channel, times that channel's weights."""

    def __init__(self, cin, cout, k, stride=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = relu

    def forward(self, x):
        y = self.bn(self.conv(x))
        return F.relu(y) if self.relu else y


# %% [stem] Stem
class Stem(nn.Module):
    """A 7 × 7 convolution with stride 2 (224 → 112 pixels, 3 → 64 channels), then 3 × 3 max pooling with stride 2
    (112 → 56): each pooled pixel keeps the largest of nine neighbours, per channel."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.conv = ConvBN(3, c.stem_channels, 7, stride=2)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.conv(x))


# %% [bottleneck] Bottleneck block and its skip connection
class Bottleneck(nn.Module):
    """F(x): a 1 × 1 convolution down to width / 4 channels, a 3 × 3 convolution there (stride 2 in the first block of
    stages 2–4), a 1 × 1 convolution back up to width. The shortcut is x itself, or a strided 1 × 1 convolution where
    the shape changes. output = ReLU(F(x) + shortcut(x)): the block learns what to add to its input."""

    def __init__(self, cin, cout, stride=1, c: Config = Config()):
        super().__init__()
        mid = cout // c.reduction
        self.reduce = ConvBN(cin, mid, 1)
        self.conv = ConvBN(mid, mid, 3, stride)
        self.expand = ConvBN(mid, cout, 1, relu=False)
        projected = cin != cout or stride != 1
        self.shortcut = ConvBN(cin, cout, 1, stride, relu=False) if projected else nn.Identity()

    def forward(self, x):
        f = self.expand(self.conv(self.reduce(x)))  # F(x)
        return F.relu(f + self.shortcut(x))


# %% [receptive-field] Receptive field
def receptive_fields(c: Config = Config()):
    """How many input pixels one output pixel can see, layer by layer. A k × k layer adds (k − 1) × jump pixels, where
    jump, the input pixels between two neighbouring outputs, multiplies by every stride. Inside a block only the 3 × 3
    convolution widens it. Returns (layer, size, jump) for the stem and every block."""
    layers = [("stem 7×7 conv", 7, 2), ("stem 3×3 max pool", 3, 2)]
    for s, depth in enumerate(c.depths):
        for b in range(depth):
            layers.append((f"stage {s + 1} block {b + 1}", 3, 2 if s > 0 and b == 0 else 1))
    size, jump, out = 1, 1, []
    for name, k, stride in layers:
        size += (k - 1) * jump
        jump *= stride
        out.append((name, size, jump))
    return out


# %% [head] Pooling and the classifier
class Head(nn.Module):
    """The mean of each of the 2,048 channels over the 7 × 7 grid, then one linear layer: 1,000 class logits."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.fc = nn.Linear(c.widths[-1], c.num_classes)

    def forward(self, x):
        pooled = x.mean((2, 3))  # (batch, 2048)
        return pooled, self.fc(pooled)


# %% [model] ResNet-50
class ResNet50(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.stem = Stem(c)
        stages, cin = [], c.stem_channels
        for s, (width, depth) in enumerate(zip(c.widths, c.depths)):
            first = Bottleneck(cin, width, stride=2 if s > 0 else 1, c=c)
            stages.append(nn.Sequential(first, *[Bottleneck(width, width, c=c) for _ in range(depth - 1)]))
            cin = width
        self.stages = nn.ModuleList(stages)
        self.head = Head(c)

    def forward(self, x):  # x: (batch, 3, 224, 224)
        out = {"stem": self.stem(x)}  # (batch, 64, 56, 56)
        h = out["stem"]
        for s, stage in enumerate(self.stages, 1):
            h = stage(h)  # stage 1: 256 × 56 × 56, 2: 512 × 28 × 28, 3: 1024 × 14 × 14, 4: 2048 × 7 × 7
            out[f"stage{s}"] = h
        out["pooled"], out["logits"] = self.head(h)
        return out
`;export{e as default};
