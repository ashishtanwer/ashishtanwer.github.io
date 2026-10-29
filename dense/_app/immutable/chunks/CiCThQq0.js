const e=`# Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0), the latent text-to-image U-Net of the SDXL tab, written
# as plain PyTorch blocks: two CLIP text encoders, the size and crop micro-conditioning, the three-level U-Net with
# 1 × 2 × 10 transformer layers, classifier-free guidance, the Euler sampler and the VAE. The U-Net works on a
# 4 × H/8 × W/8 latent; the VAE maps it to pixels. Each "# %% [id]" section is one step of the tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 4
    widths: tuple = (320, 640, 1280)  # U-Net channels at latent sizes H/8, H/16, H/32
    depth: tuple = (0, 2, 10)  # transformer layers in each transformer block, per level: none at the first
    heads: tuple = (5, 10, 20)  # head width 64 wherever there is attention
    layers_per_level: int = 2
    groups: int = 32
    eps: float = 1e-5
    # The two text encoders: CLIP ViT-L/14 and OpenCLIP ViT-bigG/14, both with CLIP's tokenizer.
    text_dims: tuple = (768, 1280)
    text_layers: tuple = (12, 32)
    text_heads: tuple = (12, 20)
    text_mlp: tuple = (3072, 5120)
    text_act: tuple = ("quick_gelu", "gelu")
    pooled_dim: int = 1280  # bigG's projection of its end token
    vocab: int = 49408
    max_tokens: int = 77
    size_dim: int = 256  # sinusoid width of each micro-conditioning number
    vae_widths: tuple = (128, 256, 512, 512)  # the VAE's channels at H, H/2, H/4, H/8
    vae_scale: float = 0.13025  # SDXL's VAE latents × 0.13025 have about unit variance
    timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012

    @property
    def text_dim(self):  # the concatenated width every cross-attention reads: 768 + 1280 = 2048
        return sum(self.text_dims)


# %% [text-encoders] Two text encoders
class TextEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer layer with a causal mask: each token sees only the ones before it."""

    def __init__(self, d, heads, mlp, act):
        super().__init__()
        self.heads, self.act = heads, act
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, mlp), nn.Linear(mlp, d)

    def forward(self, x):  # x: (batch, 77, d)
        b, n, d = x.shape
        h = self.norm1(x)
        q, k, v = (proj(h).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        future = torch.ones(n, n, dtype=torch.bool).triu(1)
        scores = (q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)).masked_fill(future, float("-inf"))
        x = x + self.out((scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d))
        h = self.fc1(self.norm2(x))
        h = h * torch.sigmoid(1.702 * h) if self.act == "quick_gelu" else F.gelu(h)
        return x + self.fc2(h)


class TextEncoder(nn.Module):
    """Token ids (padded to 77) → the second-to-last layer's output at every position (what SDXL's cross-attention
    reads), and the end token after the last layer and LayerNorm, projected when the encoder has a projection."""

    def __init__(self, c: Config, which: int, projection: bool):
        super().__init__()
        d = c.text_dims[which]
        self.token_embedding = nn.Embedding(c.vocab, d)
        self.position_embedding = nn.Embedding(c.max_tokens, d)
        self.layers = nn.ModuleList(
            TextEncoderLayer(d, c.text_heads[which], c.text_mlp[which], c.text_act[which])
            for _ in range(c.text_layers[which]))
        self.final_norm = nn.LayerNorm(d)
        self.projection = nn.Linear(d, c.pooled_dim, bias=False) if projection else None

    def forward(self, ids):  # ids: (batch, 77)
        x = self.token_embedding(ids) + self.position_embedding.weight[: ids.shape[1]]
        for layer in self.layers[:-1]:
            x = layer(x)
        penultimate = x  # (batch, 77, d): the last layer is skipped for the per-token vectors
        x = self.final_norm(self.layers[-1](x))
        end = x[torch.arange(len(ids)), ids.argmax(-1)]  # the end token has the largest id in CLIP's vocabulary
        return penultimate, (end if self.projection is None else self.projection(end))


def encode(clip_l, clip_g, ids_l, ids_g):
    """Both encoders read the same prompt; their per-token vectors are concatenated (77 × 2048) and bigG's projected
    end token is the pooled vector (1280)."""
    tokens_l, _ = clip_l(ids_l)
    tokens_g, pooled = clip_g(ids_g)
    return torch.cat([tokens_l, tokens_g], dim=-1), pooled


# %% [timestep-embedding] Timestep embedding
def sinusoid(t, dim):
    """cos then sin of t at dim/2 frequencies from 1 down to 1/10000."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half)
    angles = t[:, None].float() * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class TimestepEmbedding(nn.Module):
    """t → sinusoid (320) → Linear, SiLU, Linear → temb (1280)."""

    def __init__(self, width: int, out: int):
        super().__init__()
        self.width = width
        self.linear_1 = nn.Linear(width, out)
        self.linear_2 = nn.Linear(out, out)

    def forward(self, t):
        return self.linear_2(F.silu(self.linear_1(sinusoid(t, self.width))))


# %% [added-conditioning] Size and crop conditioning
class AddedConditioning(nn.Module):
    """Micro-conditioning: six numbers (the training image's original height and width, the crop's top and left,
    the target height and width), each as a 256-wide sinusoid, next to the pooled text vector → Linear, SiLU,
    Linear → a 1280 vector added to temb. Every ResNet block reads the sum."""

    def __init__(self, c: Config):
        super().__init__()
        self.size_dim = c.size_dim
        self.linear_1 = nn.Linear(c.pooled_dim + 6 * c.size_dim, 4 * c.widths[0])  # 1280 + 1536 = 2816 → 1280
        self.linear_2 = nn.Linear(4 * c.widths[0], 4 * c.widths[0])

    def forward(self, sizes, pooled):  # sizes: (batch, 6), pooled: (batch, 1280)
        waves = sinusoid(sizes.flatten(), self.size_dim).reshape(len(sizes), -1)  # (batch, 6 · 256)
        return self.linear_2(F.silu(self.linear_1(torch.cat([pooled, waves], dim=1))))


# %% [resnet] ResNet block with the timestep
class ResnetBlock(nn.Module):
    """Two 3 × 3 convolutions with a skip around them; the embedding (if any) is added between them as one
    number per channel. The VAE uses the same block without it."""

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


# %% [transformer] Transformer block: a stack of layers over the pixels
class Attention(nn.Module):
    """Multi-head attention; keys and values come from \`context\` (the text) when it is given."""

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


class TransformerLayer(nn.Module):
    """Self-attention over the pixels, cross-attention to the text, GEGLU feed-forward; each after a LayerNorm,
    each added back."""

    def __init__(self, width, heads, text_dim):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attention = Attention(width, heads)
        self.cross_attention = Attention(width, heads, context_dim=text_dim)
        self.ff_in = nn.Linear(width, 2 * 4 * width)  # GEGLU: a value half and a gate half
        self.ff_out = nn.Linear(4 * width, width)

    def forward(self, tokens, text):
        tokens = tokens + self.self_attention(self.norm1(tokens))
        tokens = tokens + self.cross_attention(self.norm2(tokens), text)
        value, gate = self.ff_in(self.norm3(tokens)).chunk(2, dim=-1)
        return tokens + self.ff_out(value * F.gelu(gate))


class TransformerBlock(nn.Module):
    """Feature map → GroupNorm → one token per pixel → Linear → \`depth\` layers → Linear → feature map, added to
    the input. SDXL stacks 2 layers at H/16 and 10 at H/32."""

    def __init__(self, width: int, heads: int, depth: int, c: Config):
        super().__init__()
        self.norm = nn.GroupNorm(c.groups, width, eps=1e-6)
        self.proj_in = nn.Linear(width, width)
        self.layers = nn.ModuleList(TransformerLayer(width, heads, c.text_dim) for _ in range(depth))
        self.proj_out = nn.Linear(width, width)

    def forward(self, x, text):  # x: (batch, width, h, w), text: (batch, 77, 2048)
        b, width, h, w = x.shape
        tokens = self.proj_in(self.norm(x).flatten(2).transpose(1, 2))  # (batch, h·w, width)
        for layer in self.layers:
            tokens = layer(tokens, text)
        return x + self.proj_out(tokens).transpose(1, 2).reshape(b, width, h, w)


# %% [unet] U-Net levels and skips
class Level(nn.Module):
    def __init__(self, resnets, attentions, resample=None):
        super().__init__()
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)  # transformer blocks (U-Net) or self-attention (VAE), or none
        self.resample = resample


class UNet(nn.Module):
    """Noisy latent, timestep, text, pooled text and sizes → predicted noise. Down: each level halves the latent and
    saves every block's output. Up: each block concatenates the matching saved output to its input."""

    def __init__(self, c: Config):
        super().__init__()
        w, n, temb = c.widths, c.layers_per_level, 4 * c.widths[0]
        res = lambda c_in, c_out: ResnetBlock(c_in, c_out, temb, c.groups, c.eps)
        text = lambda i: [TransformerBlock(w[i], c.heads[i], c.depth[i], c) for _ in range(n)] if c.depth[i] else []
        self.time_embedding = TimestepEmbedding(w[0], temb)
        self.add_embedding = AddedConditioning(c)
        self.conv_in = nn.Conv2d(c.latent_channels, w[0], 3, padding=1)
        self.down = nn.ModuleList()
        for i, width in enumerate(w):
            down = nn.Conv2d(width, width, 3, stride=2, padding=1) if i < len(w) - 1 else None
            self.down.append(Level([res(w[max(i - 1, 0)] if j == 0 else width, width) for j in range(n)], text(i), down))
        self.mid = Level([res(w[-1], w[-1]) for _ in range(2)], [TransformerBlock(w[-1], c.heads[-1], c.depth[-1], c)])
        skip_widths = [w[0]] + [width for i, width in enumerate(w) for _ in range(n + (i < len(w) - 1))]
        self.up = nn.ModuleList()
        width_in = w[-1]
        for i, width in enumerate(reversed(w)):
            level = len(w) - 1 - i
            resnets = [res((width_in if j == 0 else width) + skip_widths.pop(), width) for j in range(n + 1)]
            blocks = [TransformerBlock(width, c.heads[level], c.depth[level], c) for _ in range(n + 1)] if c.depth[level] else []
            self.up.append(Level(resnets, blocks, nn.Conv2d(width, width, 3, padding=1) if level > 0 else None))
            width_in = width
        self.norm_out = nn.GroupNorm(c.groups, w[0], eps=c.eps)
        self.conv_out = nn.Conv2d(w[0], c.latent_channels, 3, padding=1)

    def forward(self, z, t, text, pooled, sizes):  # z: (batch, 4, h, w), text: (batch, 77, 2048), sizes: (batch, 6)
        temb = self.time_embedding(t) + self.add_embedding(sizes, pooled)
        h = self.conv_in(z)
        skips = [h]
        for level in self.down:
            for j, resnet in enumerate(level.resnets):
                h = resnet(h, temb)
                if len(level.attentions):
                    h = level.attentions[j](h, text)
                skips.append(h)
            if level.resample is not None:
                h = level.resample(h)  # stride 2: 64 → 32 → 16
                skips.append(h)
        h = self.mid.resnets[1](self.mid.attentions[0](self.mid.resnets[0](h, temb), text), temb)
        for level in self.up:
            for j, resnet in enumerate(level.resnets):
                h = resnet(torch.cat([h, skips.pop()], dim=1), temb)
                if len(level.attentions):
                    h = level.attentions[j](h, text)
            if level.resample is not None:
                h = level.resample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.conv_out(F.silu(self.norm_out(h)))  # ε̂: (batch, 4, h, w)


# %% [vae] VAE: pixels ↔ latents
class SelfAttention(nn.Module):
    """One head over every pixel, as at the bottom of the VAE."""

    def __init__(self, width: int, groups: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, width, eps=1e-6)
        self.to_q, self.to_k, self.to_v, self.to_out = (nn.Linear(width, width) for _ in range(4))

    def forward(self, x):
        b, width, h, w = x.shape
        tokens = self.group_norm(x).flatten(2).transpose(1, 2)
        weights = torch.softmax(self.to_q(tokens) @ self.to_k(tokens).transpose(1, 2) / math.sqrt(width), -1)
        return x + self.to_out(weights @ self.to_v(tokens)).transpose(1, 2).reshape(b, width, h, w)


class VAE(nn.Module):
    """Encoder: 3 × H × W pixels → mean and log-variance of a 4 × H/8 × W/8 latent. Decoder: the mirror image.
    The same layout as SD 1.5's VAE, retrained for SDXL (hence the different scale)."""

    def __init__(self, c: Config):
        super().__init__()
        w, g, n = c.vae_widths, c.groups, c.layers_per_level
        res = lambda c_in, c_out: ResnetBlock(c_in, c_out, None, g, 1e-6)
        self.enc_in = nn.Conv2d(3, w[0], 3, padding=1)
        self.enc_down = nn.ModuleList(
            Level([res(w[max(i - 1, 0)] if j == 0 else width, width) for j in range(n)], [],
                  nn.Conv2d(width, width, 3, stride=2) if i < len(w) - 1 else None)
            for i, width in enumerate(w))
        self.enc_mid = Level([res(w[-1], w[-1]) for _ in range(2)], [SelfAttention(w[-1], g)])
        self.enc_norm = nn.GroupNorm(g, w[-1], eps=1e-6)
        self.enc_out = nn.Conv2d(w[-1], 2 * c.latent_channels, 3, padding=1)
        self.quant = nn.Conv2d(2 * c.latent_channels, 2 * c.latent_channels, 1)
        self.post_quant = nn.Conv2d(c.latent_channels, c.latent_channels, 1)
        self.dec_in = nn.Conv2d(c.latent_channels, w[-1], 3, padding=1)
        self.dec_mid = Level([res(w[-1], w[-1]) for _ in range(2)], [SelfAttention(w[-1], g)])
        up, width_in = [], w[-1]
        for i, width in enumerate(reversed(w)):
            up.append(Level([res(width_in if j == 0 else width, width) for j in range(n + 1)], [],
                            nn.Conv2d(width, width, 3, padding=1) if i < len(w) - 1 else None))
            width_in = width
        self.dec_up = nn.ModuleList(up)
        self.dec_norm = nn.GroupNorm(g, w[0], eps=1e-6)
        self.dec_out = nn.Conv2d(w[0], 3, 3, padding=1)

    @staticmethod
    def middle(level, h):
        return level.resnets[1](level.attentions[0](level.resnets[0](h)))

    def encode(self, x):  # x: (batch, 3, H, W) in [-1, 1] → mean, log-variance (batch, 4, H/8, W/8)
        h = self.enc_in(x)
        for level in self.enc_down:
            for resnet in level.resnets:
                h = resnet(h)
            if level.resample is not None:
                h = level.resample(F.pad(h, (0, 1, 0, 1)))
        h = self.enc_out(F.silu(self.enc_norm(self.middle(self.enc_mid, h))))
        mean, logvar = self.quant(h).chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)

    def decode(self, z):  # z: (batch, 4, H/8, W/8), unscaled → (batch, 3, H, W) in about [-1, 1]
        h = self.middle(self.dec_mid, self.dec_in(self.post_quant(z)))
        for level in self.dec_up:
            for resnet in level.resnets:
                h = resnet(h)
            if level.resample is not None:
                h = level.resample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.dec_out(F.silu(self.dec_norm(h)))


# %% [guidance] Classifier-free guidance
def guide(eps_empty, eps_prompt, scale=5.0):
    """ε̂ = ε(∅) + s · (ε(c) − ε(∅)). SDXL's ∅ is all zeros: with no negative prompt the pipeline zeroes both the
    token vectors and the pooled vector instead of encoding an empty string."""
    return eps_empty + scale * (eps_prompt - eps_empty)


# %% [euler-step] Euler step
def noise_schedule(c: Config):
    """"Scaled linear": √β grows linearly, from √0.00085 to √0.012."""
    betas = torch.linspace(c.beta_start**0.5, c.beta_end**0.5, c.timesteps) ** 2
    return torch.cumprod(1 - betas, dim=0)  # ᾱ_t


def euler_schedule(steps, c: Config):
    """Every 1000/steps-th timestep, offset by one (951, 901, …, 1 for 20 steps), and its noise level
    σ = √((1 − ᾱ) / ᾱ), then σ = 0."""
    alphas_cumprod = noise_schedule(c)
    timesteps = torch.arange(steps).flip(0) * (c.timesteps // steps) + 1
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    return timesteps, torch.cat([sigmas[timesteps], torch.zeros(1)])


def euler_step(z, eps, sigma, sigma_next):
    """Here z = x_0 + σ · ε (the U-Net reads z / √(σ² + 1)). One Euler step of dz/dσ = ε̂ from σ to σ_next."""
    x0 = z - sigma * eps
    return z + (sigma_next - sigma) * (z - x0) / sigma, x0


# %% [model] Text to image
@torch.no_grad()
def sample(unet, vae, text, pooled, c: Config, generator, size=512, steps=20, scale=5.0, sizes=None):
    """Euler with classifier-free guidance: the zeroed ∅ and the prompt in one batch, then the VAE decodes the
    final latent. \`text\` (1, 77, 2048) and \`pooled\` (1, 1280) come from \`encode\`; \`sizes\` defaults to an
    unscaled, uncropped image of the output size."""
    sizes = torch.tensor([[size, size, 0, 0, size, size]]) if sizes is None else sizes
    latent = size // 2 ** (len(c.vae_widths) - 1)  # 512 px → 64
    timesteps, sigmas = euler_schedule(steps, c)
    z = torch.randn(1, c.latent_channels, latent, latent, generator=generator) * (sigmas[0] ** 2 + 1) ** 0.5
    text, pooled = torch.cat([torch.zeros_like(text), text]), torch.cat([torch.zeros_like(pooled), pooled])
    for i, t in enumerate(timesteps):
        z_in = torch.cat([z, z]) / (sigmas[i] ** 2 + 1) ** 0.5
        eps_empty, eps_prompt = unet(z_in, t.repeat(2), text, pooled, sizes.repeat(2, 1)).chunk(2)
        z, _ = euler_step(z, guide(eps_empty, eps_prompt, scale), sigmas[i], sigmas[i + 1])
    return (vae.decode(z / c.vae_scale) / 2 + 0.5).clamp(0, 1)
`;export{e as default};
