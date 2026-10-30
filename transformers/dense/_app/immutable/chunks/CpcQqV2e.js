const e=`# ControlNet (lllyasviel/sd-controlnet-canny) on Stable Diffusion 1.5, the model of the ControlNet tab, as plain PyTorch:
# the canny edges of a photo steer SD 1.5's frozen U-Net through a trainable copy of its encoder, whose outputs are added
# to the U-Net's skip connections and middle block through zero-initialized 1 × 1 convolutions.
# The U-Net, its blocks, the VAE, guidance and the DDIM step are the SD 1.5 tab's (pytorch/diffusion_sd15.py).
# Each "# %% [id]" section is one step of the tab's diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_sd15 import Config as SD15Config
from diffusion_sd15 import UNet, ddim_step, ddim_timesteps, guide, noise_schedule


@dataclass
class Config(SD15Config):
    condition_widths: tuple = (16, 32, 96, 256)  # the edge image's encoder: 512 → 256 → 128 → 64 px (here 256 → 32)
    low: int = 100  # canny thresholds on the gradient's |dx| + |dy|, as the model was trained
    high: int = 200


# %% [canny] Canny edges
def sobel(image):
    """3 × 3 Sobel derivatives of a (channels, H, W) image with values 0–255, edges replicated. Each pixel keeps
    the channel whose gradient |dx| + |dy| is largest (the first one on a tie)."""
    kernel = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)
    c = image.shape[0]
    padded = F.pad(image.double()[None], (1, 1, 1, 1), mode="replicate")
    dx = F.conv2d(padded, kernel.expand(c, 1, 3, 3), groups=c)[0].long()
    dy = F.conv2d(padded, kernel.T.expand(c, 1, 3, 3), groups=c)[0].long()
    magnitude = dx.abs() + dy.abs()
    best = magnitude.argmax(0, keepdim=True)
    return dx.gather(0, best)[0], dy.gather(0, best)[0], magnitude.gather(0, best)[0]


def thin(dx, dy, magnitude):
    """Non-maximum suppression: keep a pixel only if it beats both neighbours along its gradient, binned to
    horizontal, vertical or one of the diagonals by integer tangents of 22.5° and 67.5°."""
    h, w = magnitude.shape
    m = F.pad(magnitude, (1, 1, 1, 1))  # 0 outside the image
    at = lambda y, x: m[1 + y : 1 + y + h, 1 + x : 1 + x + w]
    ax, ay = dx.abs(), dy.abs() << 15
    tan22 = ax * 13573  # tan(22.5°) · 2¹⁵
    tan67 = tan22 + (ax << 16)  # tan(67.5°) · 2¹⁵ = (tan(22.5°) + 2) · 2¹⁵
    horizontal = (magnitude > at(0, -1)) & (magnitude >= at(0, 1))
    vertical = (magnitude > at(-1, 0)) & (magnitude >= at(1, 0))
    same_sign = (dx ^ dy) >= 0
    diagonal = torch.where(same_sign, (magnitude > at(-1, -1)) & (magnitude > at(1, 1)),
                           (magnitude > at(-1, 1)) & (magnitude > at(1, -1)))
    return torch.where(ay < tan22, horizontal, torch.where(ay > tan67, vertical, diagonal))


def hysteresis(peaks, magnitude, low, high):
    """Strong edges (> high) are kept; weak ones (> low) only if a chain of weak edges connects them to a strong one."""
    weak = peaks & (magnitude > low)
    edges = weak & (magnitude > high)
    while True:  # grow the strong edges into touching weak ones (8 neighbours) until nothing changes
        grown = weak & (F.max_pool2d(edges[None].float(), 3, stride=1, padding=1)[0] > 0)
        if torch.equal(grown, edges):
            return edges
        edges = grown


def canny(image, low=100, high=200):
    """(channels, H, W) image with values 0–255 → (H, W) edge map, 255 on an edge and 0 elsewhere."""
    dx, dy, magnitude = sobel(image)
    return hysteresis(thin(dx, dy, magnitude), magnitude, low, high).to(torch.uint8) * 255


# %% [condition-embedding] Edge map → latent-sized features
class ConditionEmbedding(nn.Module):
    """The edge map, 3 × H × W in [0, 1], → 320 × H/8 × W/8, the width and size of conv_in's output: eight 3 × 3
    convolutions with SiLU between them, every second one stride 2; the last starts at zero."""

    def __init__(self, c: Config):
        super().__init__()
        widths = c.condition_widths
        self.conv_in = nn.Conv2d(3, widths[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        for width_in, width_out in zip(widths[:-1], widths[1:]):
            self.blocks.append(nn.Conv2d(width_in, width_in, 3, padding=1))
            self.blocks.append(nn.Conv2d(width_in, width_out, 3, padding=1, stride=2))
        self.conv_out = nn.Conv2d(widths[-1], c.widths[0], 3, padding=1)

    def forward(self, edges):
        h = F.silu(self.conv_in(edges))
        for conv in self.blocks:
            h = F.silu(conv(h))
        return self.conv_out(h)


# %% [control-branch] The trainable copy of the encoder
class ControlNet(nn.Module):
    """SD 1.5's encoder copied: its timestep embedding, conv_in, down levels and middle, with weights of its own.
    The edge features join right after conv_in. Every output the U-Net would save for a skip (12 here) and the
    middle's output pass through a 1 × 1 convolution of their own: these are the residuals."""

    def __init__(self, c: Config):
        super().__init__()
        encoder = UNet(c)  # the same layout as SD 1.5's U-Net; only its first half is kept
        self.time_embedding, self.conv_in = encoder.time_embedding, encoder.conv_in
        self.down, self.mid = encoder.down, encoder.mid
        self.cond_embedding = ConditionEmbedding(c)
        widths = [c.widths[0]] + [w for i, w in enumerate(c.widths) for _ in range(c.layers_per_level + (i < len(c.widths) - 1))]
        self.zero_convs = nn.ModuleList(nn.Conv2d(w, w, 1) for w in widths)  # one per skip: 320 ×4, 640 ×3, 1280 ×5
        self.zero_mid = nn.Conv2d(c.widths[-1], c.widths[-1], 1)

    def features(self, z, t, text, edges):
        """The copy's own forward pass: the outputs it would save for skips, and the middle's output."""
        temb = self.time_embedding(t)
        h = self.conv_in(z) + self.cond_embedding(edges)
        saved = [h]
        for level in self.down:
            for j, resnet in enumerate(level.resnets):
                h = resnet(h, temb)
                if len(level.attentions):
                    h = level.attentions[j](h, text)
                saved.append(h)
            if level.resample is not None:
                h = level.resample(h)
                saved.append(h)
        h = self.mid.resnets[1](self.mid.attentions[0](self.mid.resnets[0](h, temb), text), temb)
        return saved, h

    def forward(self, z, t, text, edges):  # → 12 down residuals and the middle's residual
        saved, middle = self.features(z, t, text, edges)
        return [conv(s) for conv, s in zip(self.zero_convs, saved)], self.zero_mid(middle)


# %% [zero-convolution] Zero convolutions: training starts from SD 1.5 itself
def control_from_unet(unet: UNet, c: Config) -> ControlNet:
    """A fresh ControlNet: the encoder's weights copied from the U-Net, and every 1 × 1 convolution (and the edge
    encoder's last one) set to zero. Its residuals are then exactly zero, so the controlled U-Net starts out as
    SD 1.5 unchanged; training moves the zero convolutions away from zero only as far as the edges help."""
    control = ControlNet(c)
    for name in ("time_embedding", "conv_in", "down", "mid"):
        getattr(control, name).load_state_dict(getattr(unet, name).state_dict())
    for conv in [*control.zero_convs, control.zero_mid, control.cond_embedding.conv_out]:
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
    return control


# %% [residuals] The U-Net with the residuals in its skips
class ControlledUNet(UNet):
    """SD 1.5's U-Net, frozen. Each saved skip gets its residual added before the up path reads it, and the
    middle block's output gets the middle residual."""

    def forward(self, z, t, text, down_residuals=None, mid_residual=None):
        temb = self.time_embedding(t)
        h = self.conv_in(z)
        skips = [h]
        for level in self.down:
            for j, resnet in enumerate(level.resnets):
                h = resnet(h, temb)
                if len(level.attentions):
                    h = level.attentions[j](h, text)
                skips.append(h)
            if level.resample is not None:
                h = level.resample(h)
                skips.append(h)
        if down_residuals is not None:
            skips = [skip + residual for skip, residual in zip(skips, down_residuals)]
        h = self.mid.resnets[1](self.mid.attentions[0](self.mid.resnets[0](h, temb), text), temb)
        if mid_residual is not None:
            h = h + mid_residual
        for level in self.up:
            for j, resnet in enumerate(level.resnets):
                h = resnet(torch.cat([h, skips.pop()], dim=1), temb)
                if len(level.attentions):
                    h = level.attentions[j](h, text)
            if level.resample is not None:
                h = level.resample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.conv_out(F.silu(self.norm_out(h)))


# %% [conditioning-scale] Conditioning scale: one controlled call
def controlled_eps(unet, control, z, t, text, edges, scale=1.0):
    """ε̂ with control: every residual is multiplied by \`scale\` before it is added. 0 is SD 1.5 itself, 1 is how
    the ControlNet was trained, and in between the edges are followed more loosely."""
    down, mid = control(z, t, text, edges)
    return unet(z, t, text, [scale * r for r in down], scale * mid)


# %% [guidance] Guidance with control in both passes
def guided_eps(unet, control, z, t, empty, prompt, edges, scale=1.0, guidance=7.5):
    """Classifier-free guidance, as in SD 1.5. The edges steer both the empty-prompt and the prompt pass, so the
    difference that guidance amplifies is the prompt's alone."""
    text = torch.cat([empty, prompt])
    eps_empty, eps_prompt = controlled_eps(unet, control, torch.cat([z, z]), t.repeat(2), text,
                                           torch.cat([edges, edges]), scale).chunk(2)
    return guide(eps_empty, eps_prompt, guidance)


# %% [model] Edges to image
@torch.no_grad()
def sample(unet, control, vae, photo, prompt, empty, c: Config, generator, size=256, steps=20, scale=1.0, guidance=7.5):
    """Canny edges of \`photo\` (3 × size × size, values 0–255) steer DDIM with guidance; the VAE decodes the result.
    \`prompt\` and \`empty\` are the CLIP text encoder's outputs, (1, 77, 768) each."""
    edges = canny(photo, c.low, c.high).float().div(255).expand(1, 3, -1, -1)  # one channel repeated as RGB
    alphas_cumprod = noise_schedule(c)
    latent = size // 2 ** (len(c.vae_widths) - 1)
    z = torch.randn(1, c.latent_channels, latent, latent, generator=generator)
    for t in ddim_timesteps(steps, c):
        z, _ = ddim_step(z, guided_eps(unet, control, z, t, empty, prompt, edges, scale, guidance), t, alphas_cumprod, steps)
    return (vae.decode(z / c.vae_scale) / 2 + 0.5).clamp(0, 1)
`;export{e as default};
