const e=`# What GoogLeNet's channels respond to, the cnn-features tab, as plain PyTorch.
# Three ways of looking inside a trained image classifier: feature visualization (optimize an input image until one
# channel fires as strongly as it can; Olah, Mordvintsev and Schubert's Distill article of 2017, with Lucid's
# regularizers), the dataset images that make the same channel fire most, and Grad-CAM (Selvaraju et al. 2016,
# arXiv:1610.02391), which weights the last convolutional layer's channels by the gradient of one class's score to
# show where the network looked for that class. Weights: torchvision's GoogLeNet IMAGENET1K_V1 (Szegedy et al. 2014,
# arXiv:1409.4842), which torchvision says are ported from the original paper's model.
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class Config:
    image_size: int = 224
    resize: int = 256  # shortest side to 256 (bilinear), then the centre 224 × 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    bn_eps: float = 1e-3  # the ported TensorFlow weights' BatchNorm epsilon
    # Feature visualization, Lucid's defaults (lucid/optvis): Fourier parameters with a 1/f spectrum, decorrelated
    # colour, sigmoid; Adam at 0.05 for 512 steps; the transforms of \`standard_transforms\`.
    steps: int = 512
    lr: float = 0.05
    init_sd: float = 0.01
    raw_sd: float = 1.0  # raw pixels start from visible noise: a flat grey start leaves deep channels at 0, no gradient
    decay_power: float = 1.0
    magic: float = 4.0  # Lucid divides the inverse FFT by 4 ("reduces saturation")
    pad: int = 12
    jitter: tuple = (8, 4)
    scales: tuple = tuple(1 + (i - 5) / 50 for i in range(11))
    angles: tuple = tuple(range(-10, 11)) + 5 * (0,)


# GoogLeNet's inception blocks as torchvision builds them: (1×1, 3×3 reduce, 3×3, "5×5" reduce, "5×5", pool proj).
# torchvision's "5×5" branch is a 3 × 3 convolution, as in the TensorFlow model the weights were ported from.
INCEPTION = {
    "inception3a": (64, 96, 128, 16, 32, 32), "inception3b": (128, 128, 192, 32, 96, 64),
    "inception4a": (192, 96, 208, 16, 48, 64), "inception4b": (160, 112, 224, 24, 64, 64),
    "inception4c": (128, 128, 256, 24, 64, 64), "inception4d": (112, 144, 288, 32, 64, 64),
    "inception4e": (256, 160, 320, 32, 128, 128), "inception5a": (256, 160, 320, 32, 128, 128),
    "inception5b": (384, 192, 384, 48, 128, 128),
}


# %% [preprocess] The image the network sees
def preprocess(image: Image.Image, c: Config = Config()):
    """PIL image → (1, 3, 224, 224) in 0 … 1: shortest side to 256 (PIL bilinear, as torchvision's Resize), the centre
    224 × 224. normalize() then applies ImageNet's mean and standard deviation."""
    w, h = image.size
    if w <= h:
        size = (c.resize, int(c.resize * h / w))
    else:
        size = (int(c.resize * w / h), c.resize)
    image = image.convert("RGB").resize(size, Image.BILINEAR)
    left = int(round((size[0] - c.image_size) / 2.0))
    top = int(round((size[1] - c.image_size) / 2.0))
    image = image.crop((left, top, left + c.image_size, top + c.image_size))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]


def normalize(x, c: Config = Config()):
    mean = torch.tensor(c.mean, device=x.device)[:, None, None]
    std = torch.tensor(c.std, device=x.device)[:, None, None]
    return (x - mean) / std


# %% [model] GoogLeNet, layer by layer
def basic(x, sd, name, c: Config = Config(), stride=1, padding=0):
    """torchvision's BasicConv2d: convolution without bias, BatchNorm (running statistics), ReLU."""
    x = F.conv2d(x, sd[f"{name}.conv.weight"], stride=stride, padding=padding)
    x = F.batch_norm(x, sd[f"{name}.bn.running_mean"], sd[f"{name}.bn.running_var"], sd[f"{name}.bn.weight"],
                     sd[f"{name}.bn.bias"], eps=c.bn_eps)
    return F.relu(x)


def inception(x, sd, name, c: Config = Config()):
    """Four branches side by side: 1×1; 1×1 → 3×3; 1×1 → 3×3; 3×3 max pool → 1×1. Output: their channels concatenated."""
    b1 = basic(x, sd, f"{name}.branch1", c)
    b2 = basic(basic(x, sd, f"{name}.branch2.0", c), sd, f"{name}.branch2.1", c, padding=1)
    b3 = basic(basic(x, sd, f"{name}.branch3.0", c), sd, f"{name}.branch3.1", c, padding=1)
    b4 = basic(F.max_pool2d(x, 3, 1, 1, ceil_mode=True), sd, f"{name}.branch4.1", c)
    return torch.cat([b1, b2, b3, b4], 1)


def googlenet(x, sd, stop=None, c: Config = Config()):
    """x: (B, 3, H, W) normalized → {layer: activation} for every stage up to \`stop\` (all, and the logits, if None).
    torchvision's transform_input first maps ImageNet normalization to the (x − 0.5) / 0.5 the weights expect."""
    std, mean = torch.tensor(c.std, device=x.device), torch.tensor(c.mean, device=x.device)
    x = x * (std / 0.5)[:, None, None] + ((mean - 0.5) / 0.5)[:, None, None]
    acts = {}
    x = acts["conv1"] = basic(x, sd, "conv1", c, stride=2, padding=3)
    x = F.max_pool2d(x, 3, 2, ceil_mode=True)
    x = acts["conv2"] = basic(x, sd, "conv2", c)
    x = acts["conv3"] = basic(x, sd, "conv3", c, padding=1)
    x = F.max_pool2d(x, 3, 2, ceil_mode=True)
    for name in INCEPTION:
        if stop in acts:
            return acts
        if name in ("inception4a", "inception5a"):
            x = F.max_pool2d(x, 3 if name == "inception4a" else 2, 2, ceil_mode=True)
        x = acts[name] = inception(x, sd, name, c)
    if stop is None:
        acts["logits"] = x.mean((2, 3)) @ sd["fc.weight"].T + sd["fc.bias"]
    return acts


def centre(layer: str, i: int) -> int:
    """Input pixel at the centre of output position i of \`layer\` (224 input): stride and offset of the max-pools."""
    j, off = {"conv1": (2, 0), "conv2": (4, 2), "conv3": (4, 2), "3": (8, 6), "4": (16, 14), "5": (32, 22)}[
        layer if layer.startswith("conv") else layer[9]]
    return j * i + off


# %% [param] An image made of Fourier coefficients
def frequencies(h: int, w: int):
    """|f| of every coefficient of a real 2D FFT of an h × w image, in cycles per pixel."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[: w // 2 + 1 + (w % 2)]
    return np.sqrt(fx * fx + fy * fy)


COLOR_SVD_SQRT = np.array([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]], dtype=np.float32)


def fourier_params(batch: int, seed: int, c: Config = Config()):
    """The optimized parameters: real and imaginary parts of each colour channel's spectrum, N(0, 0.01²), seeded."""
    shape = (batch, 3) + frequencies(c.image_size, c.image_size).shape + (2,)
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * c.init_sd


def to_image(spectrum, c: Config = Config()):
    """Spectrum → image in 0 … 1: scale coefficient f by 1 / max(|f|, 1/224) (so no frequency dominates the start),
    inverse FFT (orthonormal), ÷ 4, mix the three channels with ImageNet's colour correlation (√ of its SVD, so the
    parameters are decorrelated colours), sigmoid."""
    h = w = c.image_size
    scale = torch.tensor(1.0 / np.maximum(frequencies(h, w), 1.0 / max(h, w)) ** c.decay_power, dtype=torch.float32,
                         device=spectrum.device)
    image = torch.fft.irfft2(torch.view_as_complex(spectrum * scale[..., None]), s=(h, w), norm="ortho") / c.magic
    mix = torch.from_numpy(COLOR_SVD_SQRT / np.linalg.norm(COLOR_SVD_SQRT, axis=0).max()).to(spectrum.device)
    return torch.sigmoid(torch.einsum("bchw,dc->bdhw", image, mix))


# %% [transforms] Jitter, scale and rotate before every step
def transform(image, rng: np.random.Generator, c: Config = Config()):
    """Lucid's standard_transforms, drawn fresh each step (the same draw for the whole batch): pad 12 px of grey,
    jitter by a random crop of up to 8 px, scale by 0.9 … 1.1, rotate by −10° … 10° (zero a quarter of the time),
    jitter by up to 4 px. An image that only works at one exact position, scale and angle does not survive this."""
    x = F.pad(image, [c.pad] * 4, value=0.5)
    for k, step in enumerate(("jitter", "scale", "rotate", "jitter")):
        h, w = x.shape[-2:]
        if step == "jitter":
            d = c.jitter[0 if k == 0 else 1]
            dy, dx = rng.integers(0, d, 2)
            x = x[..., dy:dy + h - d, dx:dx + w - d]
        elif step == "scale":
            s = c.scales[rng.integers(len(c.scales))]
            x = F.interpolate(x, size=(int(np.ceil(s * h)), int(np.ceil(s * w))), mode="bilinear", align_corners=True)
        else:
            a = np.deg2rad(c.angles[rng.integers(len(c.angles))])
            theta = torch.tensor([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0]], dtype=torch.float32)
            grid = F.affine_grid(theta[None].expand(x.shape[0], 2, 3).to(x.device), list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return x


# %% [visualize] Optimize the image for one channel
def visualize(forward, layer: str, channels, seed: int, c: Config = Config(), regularize: bool = True,
              steps: int | None = None, device="cpu", keep=()):
    """One image per channel, in one batch: maximize the channel's mean activation over all positions (Lucid's
    \`channel\` objective) with Adam. forward(x) → {layer: (B, C, h, w)}. regularize=False optimizes the raw pixels
    (through a sigmoid, from Gaussian noise) with no transforms, which finds high-frequency patterns instead. Returns the final images,
    the objective at every step, and the images at the steps in \`keep\`."""
    n = len(channels)
    idx = torch.tensor(channels, device=device)
    if regularize:
        params = fourier_params(n, seed, c).to(device).requires_grad_(True)
        render = lambda: to_image(params, c)  # noqa: E731
    else:
        g = torch.Generator().manual_seed(seed)
        params = (torch.randn(n, 3, c.image_size, c.image_size, generator=g) * c.raw_sd).to(device).requires_grad_(True)
        render = lambda: torch.sigmoid(params)  # noqa: E731
    opt = torch.optim.Adam([params], lr=c.lr)
    rng = np.random.default_rng(seed)
    curve, kept = [], {}
    for step in range(steps or c.steps):
        if step in keep:
            kept[step] = render().detach().cpu()
        image = render()
        x = transform(image, rng, c) if regularize else image
        act = forward(normalize(x, c))[layer]
        value = act[torch.arange(n, device=device), idx].mean((1, 2))  # (n,)
        curve.append(value.detach().cpu())
        opt.zero_grad()
        (-value.sum()).backward()
        opt.step()
    final = render().detach().cpu()
    if steps is None and c.steps in keep:
        kept[c.steps] = final
    return final, torch.stack(curve), kept


def channel_means(forward, layer: str, channels, images, c: Config = Config()):
    """Mean activation of each channel over the positions, for images (B, 3, 224, 224) in 0 … 1: (B, len(channels))."""
    act = forward(normalize(images, c))[layer]
    return act[:, channels].mean((2, 3))


# %% [dataset] The images that make a channel fire most
def channel_maxima(act, channels):
    """act (B, C, h, w) → per image and channel the strongest position: values (B, n) and (row, col) (B, n, 2)."""
    a = act[:, channels].flatten(2)
    values, flat = a.max(-1)
    w = act.shape[-1]
    return values, torch.stack([flat // w, flat % w], -1)


def crop_box(layer: str, row: int, col: int, size: int, c: Config = Config()):
    """The size × size window of the 224 × 224 input centred on a position of \`layer\`, moved inside the image."""
    half = size // 2
    y0 = min(max(centre(layer, row) - half, 0), c.image_size - size)
    x0 = min(max(centre(layer, col) - half, 0), c.image_size - size)
    return x0, y0, x0 + size, y0 + size


# %% [gradcam] Grad-CAM: where the network looked for a class
def grad_cam(forward_split, image, classes, c: Config = Config()):
    """forward_split(x) → (A, logits): the last inception block's output A (1, 1024, 7, 7) and the logits computed
    from it. For each class: α_k = mean over positions of ∂logit/∂A_k, map = ReLU(Σ_k α_k A_k), 7 × 7. Returns
    the maps (n, 7, 7), the same upsampled to 224 × 224 (bilinear, scaled to max 1) and the logits. The input is made
    to require a gradient, so A has one even when the weights are frozen."""
    A, logits = forward_split(normalize(image.detach().requires_grad_(True), c))
    maps = []
    for k in classes:
        (grad,) = torch.autograd.grad(logits[0, k], A, retain_graph=True)
        alpha = grad.mean((2, 3))  # (1, 1024)
        maps.append(F.relu((alpha[:, :, None, None] * A).sum(1))[0])
    maps = torch.stack(maps).detach()
    up = F.interpolate(maps[:, None], size=(c.image_size, c.image_size), mode="bilinear", align_corners=False)[:, 0]
    up = up / up.flatten(1).max(1).values.clamp_min(1e-12)[:, None, None]
    return maps, up, logits.detach()
`;export{e as default};
