const e=`# Stable Video Diffusion (stabilityai/stable-video-diffusion-img2vid), the image-to-video U-Net of the SVD tab, written as
# plain PyTorch blocks. One picture conditions the clip twice: CLIP ViT-H/14 turns it into a single 1,024-vector that
# every cross-attention reads, and the VAE's latent of it is stacked onto every frame's noisy latent. Three numbers
# (frames per second, a motion bucket and the noise added to the picture) join the timestep's embedding. The U-Net is
# SD's latent U-Net with time layers interleaved: after every ResNet block a ResNet along the frames, after every
# spatial transformer block one whose tokens are the frames at each pixel, each mixed in by a learned weight. It
# predicts v under EDM's preconditioning; Euler steps on Karras sigmas from 700 down; a VAE decoder with the same
# time layers turns the latents into frames.
# Each "# %% [id]" section is one step of the tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 4
    widths: tuple = (320, 640, 1280, 1280)  # U-Net channels at latent sizes H/8, H/16, H/32, H/64
    attention_levels: tuple = (0, 1, 2)  # levels with a transformer block after each ResNet block
    layers_per_level: int = 2
    heads: tuple = (5, 10, 20, 20)  # 64 numbers per head
    groups: int = 32
    image_dim: int = 1024  # CLIP ViT-H/14's image embedding: the one key of every cross-attention
    added_dim: int = 256  # a sinusoid per micro-conditioning number
    frames: int = 14
    # CLIP ViT-H/14 image encoder
    vision_width: int = 1280
    vision_layers: int = 32
    vision_heads: int = 16
    vision_mlp: int = 5120
    patch: int = 14
    image_size: int = 224
    clip_mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    clip_std: tuple = (0.26862954, 0.26130258, 0.27577711)
    # VAE
    vae_widths: tuple = (128, 256, 512, 512)  # channels at H, H/2, H/4, H/8
    vae_scale: float = 0.18215
    # EDM sampling
    sigma_min: float = 0.002
    sigma_max: float = 700.0
    rho: float = 7.0


# %% [image-encoder] CLIP image embedding
def clip_input(pixels, c: Config):
    """The pipeline's resize to 224 × 224: a Gaussian blur with σ = (scale − 1)/2 along each axis (reflect padding),
    then bicubic interpolation; then CLIP's per-channel normalisation. pixels: (batch, 3, H, W) in [0, 1]."""
    x = pixels * 2 - 1
    height, width = x.shape[-2:]
    sigma_y, sigma_x = (max((size / c.image_size - 1) / 2, 0.001) for size in (height, width))
    size_y, size_x = (int(max(4 * s, 3)) | 1 for s in (sigma_y, sigma_x))  # odd kernel sizes
    for sigma, k, kernel_shape, pad in ((sigma_x, size_x, (1, size_x), (size_x // 2, size_x // 2, 0, 0)),
                                        (sigma_y, size_y, (size_y, 1), (0, 0, size_y // 2, size_y // 2))):
        offsets = torch.arange(k, dtype=x.dtype) - k // 2
        gauss = torch.exp(-offsets**2 / (2 * sigma**2))
        b, ch, h, w = x.shape
        padded = F.pad(x, pad, mode="reflect")
        x = F.conv2d(padded.reshape(b * ch, 1, *padded.shape[-2:]), (gauss / gauss.sum()).view(1, 1, *kernel_shape))
        x = x.view(b, ch, h, w)
    x = (F.interpolate(x, (c.image_size, c.image_size), mode="bicubic", align_corners=True) + 1) / 2
    return (x - torch.tensor(c.clip_mean)[:, None, None]) / torch.tensor(c.clip_std)[:, None, None]


class VisionLayer(nn.Module):
    """Pre-LayerNorm transformer layer over the 257 tokens (class token + 16 × 16 patches)."""

    def __init__(self, c: Config):
        super().__init__()
        d, self.heads = c.vision_width, c.vision_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, c.vision_mlp), nn.Linear(c.vision_mlp, d)

    def forward(self, x):
        b, n, d = x.shape
        h = self.norm1(x)
        q, k, v = (proj(h).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)
        x = x + self.out((weights @ v).transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ImageEncoder(nn.Module):
    """224 × 224 pixels → 14 × 14 patches + a class token → 32 layers → the class token, normalised and projected:
    one 1,024-vector for the whole picture."""

    def __init__(self, c: Config):
        super().__init__()
        d = c.vision_width
        self.patch_embedding = nn.Conv2d(3, d, c.patch, stride=c.patch, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(d))
        self.position_embedding = nn.Embedding((c.image_size // c.patch) ** 2 + 1, d)
        self.pre_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(VisionLayer(c) for _ in range(c.vision_layers))
        self.post_norm = nn.LayerNorm(d)
        self.projection = nn.Linear(d, c.image_dim, bias=False)

    def forward(self, x):  # x: clip_input(...), (batch, 3, 224, 224)
        patches = self.patch_embedding(x).flatten(2).transpose(1, 2)  # (batch, 256, 1280)
        x = torch.cat([self.class_embedding.expand(len(x), 1, -1), patches], dim=1) + self.position_embedding.weight
        x = self.pre_norm(x)
        for layer in self.layers:
            x = layer(x)
        return self.projection(self.post_norm(x[:, 0]))  # (batch, 1024)


# %% [image-latent] The picture's latent on every frame
class Encoder(nn.Module):
    """The VAE's encoder (SD's): pixels → mean and log-variance of a 4 × H/8 × W/8 latent."""

    def __init__(self, c: Config):
        super().__init__()
        w, g = c.vae_widths, c.groups
        res = lambda c_in, c_out: ResnetBlock(c_in, c_out, None, g, 1e-6)
        self.conv_in = nn.Conv2d(3, w[0], 3, padding=1)
        self.down = nn.ModuleList(
            Level([res(w[max(i - 1, 0)] if j == 0 else width, width) for j in range(c.layers_per_level)], [],
                  nn.Conv2d(width, width, 3, stride=2) if i < len(w) - 1 else None)
            for i, width in enumerate(w))
        self.mid = Level([res(w[-1], w[-1]) for _ in range(2)], [SelfAttention(w[-1], g)])
        self.norm_out = nn.GroupNorm(g, w[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(w[-1], 2 * c.latent_channels, 3, padding=1)
        self.quant = nn.Conv2d(2 * c.latent_channels, 2 * c.latent_channels, 1)

    def forward(self, x):  # x: (batch, 3, H, W) in [-1, 1]
        h = self.conv_in(x)
        for level in self.down:
            for resnet in level.resnets:
                h = resnet(h)
            if level.resample is not None:
                h = level.resample(F.pad(h, (0, 1, 0, 1)))
        h = self.mid.resnets[1](self.mid.attentions[0](self.mid.resnets[0](h)))
        mean, logvar = self.quant(self.conv_out(F.silu(self.norm_out(h)))).chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)


def image_latent(encoder, pixels, noise, augmentation=0.02):
    """The picture, with a little noise added, through the VAE's encoder: the mean, NOT multiplied by 0.18215. The
    U-Net reads it next to every frame's noisy latent: 4 + 4 = 8 input channels. pixels: (batch, 3, H, W) in [0, 1]."""
    return encoder(pixels * 2 - 1 + augmentation * noise)[0]


# %% [timestep-embedding] Timestep embedding
def sinusoid(t, dim):
    """cos then sin of t at dim/2 frequencies from 1 down to 1/10000."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half)
    angles = t[:, None].float() * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class MLP(nn.Module):
    """Linear, SiLU, Linear. The timestep's sinusoid (320) → 1,280; the U-Net's t is c_noise = ¼ ln σ, from 1.64 at
    σ = 700 down to −1.55 at σ = 0.002."""

    def __init__(self, d_in: int, inner: int, out: int):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, inner)
        self.linear_2 = nn.Linear(inner, out)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


# %% [micro-conditioning] Frame rate, motion bucket and noise augmentation
class Conditioning(nn.Module):
    """fps − 1, the motion bucket (0–255, how much moves) and the noise added to the picture each get a 256-wide
    sinusoid; side by side (768) they pass an MLP and are added to the timestep's embedding. The sum (1,280) is what
    every ResNet block reads, the same for all frames."""

    def __init__(self, c: Config):
        super().__init__()
        self.width, self.added_dim, temb = c.widths[0], c.added_dim, 4 * c.widths[0]
        self.time_embedding = MLP(c.widths[0], temb, temb)
        self.add_embedding = MLP(3 * c.added_dim, temb, temb)

    def forward(self, t, added):  # t: (batch,), added: (batch, 3) = fps − 1, motion bucket, noise augmentation
        numbers = sinusoid(added.flatten(), self.added_dim).reshape(len(added), -1)  # (batch, 768)
        return self.time_embedding(sinusoid(t, self.width)) + self.add_embedding(numbers)  # (batch, 1280)


# %% [resnet] Spatio-temporal ResNet block
class ResnetBlock(nn.Module):
    """SD's ResNet block on each frame alone: two 3 × 3 convolutions, the embedding added between them as one number
    per channel, a skip around them."""

    def __init__(self, c_in, c_out, temb_dim=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in, eps=eps)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_dim, c_out) if temb_dim else None
        self.norm2 = nn.GroupNorm(groups, c_out, eps=eps)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x, temb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_emb_proj is not None:
            h = h + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (x if self.conv_shortcut is None else self.conv_shortcut(x)) + h


class TemporalResnet(nn.Module):
    """The same block along time: 3 × 1 × 1 convolutions read each pixel in the frame before, itself and the frame
    after. Its GroupNorm statistics span all frames."""

    def __init__(self, channels, temb_dim=None, groups=32, eps=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels, eps=eps)
        self.conv1 = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))
        self.time_emb_proj = nn.Linear(temb_dim, channels) if temb_dim else None
        self.norm2 = nn.GroupNorm(groups, channels, eps=eps)
        self.conv2 = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, temb=None):  # x: (batch, C, frames, h, w), temb: (batch, frames, 1280)
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_emb_proj is not None:
            h = h + self.time_emb_proj(F.silu(temb)).transpose(1, 2)[..., None, None]
        return x + self.conv2(F.silu(self.norm2(h)))


class SpatioTemporalResnet(nn.Module):
    """Spatial block per frame, then the temporal block over its output, mixed: α · spatial + (1 − α) · temporal with
    α = sigmoid(a learned number). The VAE's decoder flips it: α = 1 − sigmoid(·)."""

    def __init__(self, c_in, c_out, temb_dim=None, groups=32, eps=1e-5, temporal_eps=None, flip=False):
        super().__init__()
        self.spatial = ResnetBlock(c_in, c_out, temb_dim, groups, eps)
        self.temporal = TemporalResnet(c_out, temb_dim, groups, temporal_eps or eps)
        self.mix_factor = nn.Parameter(torch.zeros(1))
        self.flip = flip

    def forward(self, x, temb, frames):  # x: (batch · frames, C, h, w), temb: (batch · frames, 1280) or None
        h = self.spatial(x, temb)
        bf, ch, height, width = h.shape
        video = h.view(bf // frames, frames, ch, height, width).transpose(1, 2)  # (batch, C, frames, h, w)
        moved = self.temporal(video, None if temb is None else temb.view(bf // frames, frames, -1))
        alpha = torch.sigmoid(self.mix_factor)
        alpha = 1 - alpha if self.flip else alpha
        return (alpha * video + (1 - alpha) * moved).transpose(1, 2).reshape(bf, ch, height, width)


# %% [transformer] Spatio-temporal transformer block
class Attention(nn.Module):
    """Multi-head attention; keys and values come from \`context\` when it is given."""

    def __init__(self, dim, heads, context_dim=None):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim or dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim or dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        context = x if context is None else context
        b, n, d = x.shape
        split = lambda t: t.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = split(self.to_q(x)), split(self.to_k(context)), split(self.to_v(context))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)
        return self.to_out((weights @ v).transpose(1, 2).reshape(b, n, d))


class GEGLU(nn.Module):
    """Feed-forward with a GELU gate: Linear to 2 × 4d, value × GELU(gate), Linear back."""

    def __init__(self, dim, out=None):
        super().__init__()
        self.proj = nn.Linear(dim, 2 * 4 * dim)
        self.out = nn.Linear(4 * dim, out or dim)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(value * F.gelu(gate))


class SpatialBlock(nn.Module):
    """SD's block on each frame: self-attention over the frame's pixels, cross-attention to the picture's one CLIP
    vector (a single key, so every weight is 1 and every pixel receives the same vector), GEGLU feed-forward."""

    def __init__(self, width, heads, context_dim):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attention = Attention(width, heads)
        self.cross_attention = Attention(width, heads, context_dim)
        self.ff = GEGLU(width)

    def forward(self, tokens, image):  # tokens: (batch · frames, pixels, C), image: (batch · frames, 1, 1024)
        tokens = tokens + self.self_attention(self.norm1(tokens))
        tokens = tokens + self.cross_attention(self.norm2(tokens), image)
        return tokens + self.ff(self.norm3(tokens))


class TemporalBlock(nn.Module):
    """The same at each pixel along time: the tokens are that pixel in every frame. A GEGLU first, then
    self-attention over the frames, cross-attention to the picture, GEGLU, each added back."""

    def __init__(self, width, heads, context_dim):
        super().__init__()
        self.norm_in = nn.LayerNorm(width)
        self.ff_in = GEGLU(width)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attention = Attention(width, heads)
        self.cross_attention = Attention(width, heads, context_dim)
        self.ff = GEGLU(width)

    def forward(self, tokens, image, frames):  # tokens: (batch · frames, pixels, C), image: (batch, 1, 1024)
        bf, n, ch = tokens.shape
        x = tokens.view(bf // frames, frames, n, ch).transpose(1, 2).reshape(-1, frames, ch)  # (batch · pixels, frames, C)
        x = x + self.ff_in(self.norm_in(x))
        x = x + self.self_attention(self.norm1(x))  # frames × frames weights at each pixel
        x = x + self.cross_attention(self.norm2(x), image.repeat_interleave(n, dim=0))
        x = x + self.ff(self.norm3(x))
        return x.view(bf // frames, n, frames, ch).transpose(1, 2).reshape(bf, n, ch)


class SpatioTemporalTransformer(nn.Module):
    """GroupNorm and a Linear turn each frame's map into tokens; the spatial block runs per frame; each frame's index
    (0, 1, …) gets a sinusoid and an MLP and is added before the temporal block; the two are mixed by a learned α;
    a Linear turns the tokens back into a map, added to the input."""

    def __init__(self, width, heads, c: Config):
        super().__init__()
        self.norm = nn.GroupNorm(c.groups, width, eps=1e-6)
        self.proj_in = nn.Linear(width, width)
        self.spatial = SpatialBlock(width, heads, c.image_dim)
        self.temporal = TemporalBlock(width, heads, c.image_dim)
        self.frame_embedding = MLP(width, 4 * width, width)
        self.mix_factor = nn.Parameter(torch.zeros(1))
        self.proj_out = nn.Linear(width, width)

    def forward(self, x, image, frames):  # x: (batch · frames, C, h, w), image: (batch · frames, 1, 1024)
        bf, ch, h, w = x.shape
        tokens = self.proj_in(self.norm(x).permute(0, 2, 3, 1).reshape(bf, h * w, ch))
        spatial = self.spatial(tokens, image)
        position = self.frame_embedding(sinusoid(torch.arange(frames).repeat(bf // frames), ch))[:, None]  # (b·f, 1, C)
        temporal = self.temporal(spatial + position, image[::frames], frames)  # the first frame's copy per video
        alpha = torch.sigmoid(self.mix_factor)
        tokens = alpha * spatial + (1 - alpha) * temporal
        return x + self.proj_out(tokens).reshape(bf, h, w, ch).permute(0, 3, 1, 2)


# %% [unet] U-Net levels and skips
class Level(nn.Module):
    def __init__(self, resnets, attentions, resample=None):
        super().__init__()
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)  # transformer blocks (U-Net) or self-attention (VAE), or none
        self.resample = resample


class UNet(nn.Module):
    """Noisy frames (with the picture's latent stacked on), c_noise, the picture's CLIP vector and the three numbers
    → v̂ for every frame. Frames ride along the batch axis except inside the temporal layers. Down: each level halves
    the latent and saves every block's output. Up: each block concatenates the matching saved output to its input."""

    def __init__(self, c: Config):
        super().__init__()
        w, n, temb = c.widths, c.layers_per_level, 4 * c.widths[0]
        # diffusers' block classes fix the ResNets' GroupNorm eps: 1e-6 in the levels with attention going down and in
        # every level going up, 1e-5 in the last level going down and in the middle.
        res = lambda c_in, c_out, eps=1e-6: SpatioTemporalResnet(c_in, c_out, temb, c.groups, eps)
        attend = lambda i, width: SpatioTemporalTransformer(width, c.heads[i], c)
        self.conditioning = Conditioning(c)
        self.conv_in = nn.Conv2d(2 * c.latent_channels, w[0], 3, padding=1)
        self.down = nn.ModuleList()
        for i, width in enumerate(w):
            down = nn.Conv2d(width, width, 3, stride=2, padding=1) if i < len(w) - 1 else None
            eps = 1e-6 if i in c.attention_levels else 1e-5
            self.down.append(Level([res(w[max(i - 1, 0)] if j == 0 else width, width, eps) for j in range(n)],
                                   [attend(i, width) for _ in range(n)] if i in c.attention_levels else [], down))
        self.mid = Level([res(w[-1], w[-1], 1e-5) for _ in range(2)], [attend(len(w) - 1, w[-1])])
        skip_widths = [w[0]] + [width for i, width in enumerate(w) for _ in range(n + (i < len(w) - 1))]
        self.up = nn.ModuleList()
        width_in = w[-1]
        for i, width in enumerate(reversed(w)):
            level = len(w) - 1 - i
            resnets = [res((width_in if j == 0 else width) + skip_widths.pop(), width) for j in range(n + 1)]
            attentions = [attend(level, width) for _ in range(n + 1)] if level in c.attention_levels else []
            self.up.append(Level(resnets, attentions, nn.Conv2d(width, width, 3, padding=1) if level > 0 else None))
            width_in = width
        self.norm_out = nn.GroupNorm(c.groups, w[0], eps=1e-5)
        self.conv_out = nn.Conv2d(w[0], c.latent_channels, 3, padding=1)

    def forward(self, z, t, image, added):  # z: (batch, frames, 8, h, w), t: (batch,), image: (batch, 1, 1024), added: (batch, 3)
        b, frames = z.shape[:2]
        temb = self.conditioning(t, added).repeat_interleave(frames, dim=0)  # (batch · frames, 1280)
        image = image.repeat_interleave(frames, dim=0)
        h = self.conv_in(z.flatten(0, 1))
        skips = [h]
        for level in self.down:
            for j, resnet in enumerate(level.resnets):
                h = resnet(h, temb, frames)
                if len(level.attentions):
                    h = level.attentions[j](h, image, frames)
                skips.append(h)
            if level.resample is not None:
                h = level.resample(h)
                skips.append(h)
        h = self.mid.resnets[0](h, temb, frames)
        h = self.mid.resnets[1](self.mid.attentions[0](h, image, frames), temb, frames)
        for level in self.up:
            for j, resnet in enumerate(level.resnets):
                h = resnet(torch.cat([h, skips.pop()], dim=1), temb, frames)
                if len(level.attentions):
                    h = level.attentions[j](h, image, frames)
            if level.resample is not None:
                h = level.resample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        out = self.conv_out(F.silu(self.norm_out(h)))
        return out.view(b, frames, *out.shape[1:])  # v̂: (batch, frames, 4, h, w)


# %% [guidance] Guidance that grows along the clip
def guide(out_empty, out_image, frames, low=1.0, high=3.0):
    """Classifier-free guidance with a scale per frame, from 1 at the first frame (which should stay the picture) to 3
    at the last. The 'empty' pass sees zeros instead of the CLIP vector and of the picture's latent."""
    scale = torch.linspace(low, high, frames)[None, :, None, None, None]
    return out_empty + scale * (out_image - out_empty)


# %% [edm-step] EDM preconditioning and the Euler step
def karras_sigmas(steps, c: Config):
    """Noise levels from 700 down to 0.002, evenly spaced in σ^(1/7) (dense near 0), then 0."""
    ramp = torch.linspace(0, 1, steps, dtype=torch.float64)
    low, high = c.sigma_min ** (1 / c.rho), c.sigma_max ** (1 / c.rho)
    sigmas = (high + ramp * (low - high)) ** c.rho
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float64)]).float()


def precondition(sigma):
    """EDM with σ_data = 1: the U-Net reads c_in · z and c_noise; x̂₀ = c_skip · z + c_out · v̂."""
    c_in = 1 / (sigma**2 + 1) ** 0.5
    c_skip = 1 / (sigma**2 + 1)
    c_out = -sigma / (sigma**2 + 1) ** 0.5
    c_noise = 0.25 * torch.log(sigma)
    return c_in, c_skip, c_out, c_noise


def euler_step(z, v, sigma, sigma_next):
    """x̂₀ from v̂, then one Euler step of dz/dσ = (z − x̂₀)/σ down to σ_next. No noise is added (s_churn = 0)."""
    _, c_skip, c_out, _ = precondition(sigma)
    x0 = c_skip * z + c_out * v
    return z + (sigma_next - sigma) * (z - x0) / sigma, x0


# %% [vae-decode] Temporal VAE decoder
class SelfAttention(nn.Module):
    """One head over every pixel of a frame, as at the bottom of the VAE."""

    def __init__(self, width: int, groups: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, width, eps=1e-6)
        self.to_q, self.to_k, self.to_v, self.to_out = (nn.Linear(width, width) for _ in range(4))

    def forward(self, x):
        b, width, h, w = x.shape
        tokens = self.group_norm(x).flatten(2).transpose(1, 2)
        weights = torch.softmax(self.to_q(tokens) @ self.to_k(tokens).transpose(1, 2) / math.sqrt(width), -1)
        return x + self.to_out(weights @ self.to_v(tokens)).transpose(1, 2).reshape(b, width, h, w)


class TemporalDecoder(nn.Module):
    """SD's decoder with every ResNet block made spatio-temporal (no embedding to add), and a last 3 × 1 × 1
    convolution over the frames' RGB: the frames are decoded together, so they flicker less."""

    def __init__(self, c: Config):
        super().__init__()
        w, g, n = c.vae_widths, c.groups, c.layers_per_level
        res = lambda c_in, c_out: SpatioTemporalResnet(c_in, c_out, None, g, 1e-6, 1e-5, flip=True)
        self.conv_in = nn.Conv2d(c.latent_channels, w[-1], 3, padding=1)
        self.mid = Level([res(w[-1], w[-1]) for _ in range(2)], [SelfAttention(w[-1], g)])
        up, width_in = [], w[-1]
        for i, width in enumerate(reversed(w)):
            up.append(Level([res(width_in if j == 0 else width, width) for j in range(n + 1)], [],
                            nn.Conv2d(width, width, 3, padding=1) if i < len(w) - 1 else None))
            width_in = width
        self.up = nn.ModuleList(up)
        self.norm_out = nn.GroupNorm(g, w[0], eps=1e-6)
        self.conv_out = nn.Conv2d(w[0], 3, 3, padding=1)
        self.time_conv_out = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, z, frames):  # z: (frames, 4, h, w), unscaled → (frames, 3, 8h, 8w) in about [-1, 1]
        h = self.mid.resnets[0](self.conv_in(z), None, frames)
        h = self.mid.resnets[1](self.mid.attentions[0](h), None, frames)
        for level in self.up:
            for resnet in level.resnets:
                h = resnet(h, None, frames)
            if level.resample is not None:
                h = level.resample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        rgb = self.conv_out(F.silu(self.norm_out(h)))  # (frames, 3, H, W)
        video = rgb.view(-1, frames, *rgb.shape[1:]).transpose(1, 2)  # (batch, 3, frames, H, W)
        return self.time_conv_out(video).transpose(1, 2).reshape(rgb.shape)


# %% [model] Image to video
@torch.no_grad()
def sample(unet, image_encoder, encoder, decoder, pixels, c: Config, generator, frames=14, steps=25, fps=7,
           motion_bucket=127, augmentation=0.02):
    """pixels: (1, 3, H, W) in [0, 1] → frames (frames, 3, H, W) in [0, 1]. Guidance runs both passes in one batch:
    zeros for the picture (vector and latent), then the picture."""
    embedding = image_encoder(clip_input(pixels, c))[:, None]  # (1, 1, 1024)
    noise = torch.randn(pixels.shape, generator=generator)
    latent = image_latent(encoder, pixels, noise, augmentation)  # (1, 4, H/8, W/8)
    sigmas = karras_sigmas(steps, c)
    shape = (1, frames, c.latent_channels, *latent.shape[-2:])
    z = torch.randn(shape, generator=generator) * (sigmas[0] ** 2 + 1) ** 0.5
    image = torch.cat([torch.zeros_like(embedding), embedding])
    picture = torch.cat([torch.zeros_like(latent), latent])[:, None].expand(-1, frames, -1, -1, -1)
    added = torch.tensor([[fps - 1, motion_bucket, augmentation]]).repeat(2, 1)
    for i in range(steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        c_in, _, _, c_noise = precondition(sigma)
        x = torch.cat([torch.cat([z, z]) * c_in, picture], dim=2)  # (2, frames, 8, h, w)
        out_empty, out_image = unet(x, c_noise.repeat(2), image, added).chunk(2)
        z, _ = euler_step(z, guide(out_empty, out_image, frames), sigma, sigma_next)
    video = decoder(z[0] / c.vae_scale, frames)
    return (video / 2 + 0.5).clamp(0, 1)
`;export{e as default};
