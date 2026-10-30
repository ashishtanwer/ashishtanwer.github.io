const e=`# Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5), the latent text-to-image U-Net of the SD 1.5 tab,
# written as plain PyTorch blocks: the CLIP text encoder, the VAE, the U-Net with its ResNet and transformer blocks,
# classifier-free guidance and three samplers. The U-Net works on a 4 × H/8 × W/8 latent; the VAE maps it to pixels.
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
    text_levels: tuple = (0, 1, 2)  # levels with a transformer block after each ResNet block
    layers_per_level: int = 2
    heads: int = 8  # head width = channels / 8: 40, 80, 160
    groups: int = 32
    eps: float = 1e-5
    text_dim: int = 768  # CLIP ViT-L/14's text width
    vocab: int = 49408
    text_layers: int = 12
    text_heads: int = 12
    text_mlp: int = 3072
    max_tokens: int = 77
    vae_widths: tuple = (128, 256, 512, 512)  # the VAE's channels at H, H/2, H/4, H/8
    vae_scale: float = 0.18215  # latents × 0.18215 have about unit variance
    timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012


# %% [text-encoder] CLIP text encoder
class TextEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer layer with a causal mask: each token sees only the ones before it."""

    def __init__(self, c: Config):
        super().__init__()
        d, self.heads = c.text_dim, c.text_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, c.text_mlp), nn.Linear(c.text_mlp, d)

    def forward(self, x):  # x: (batch, 77, 768)
        b, n, d = x.shape
        h = self.norm1(x)
        q, k, v = (proj(h).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        future = torch.ones(n, n, dtype=torch.bool).triu(1)
        scores = (q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)).masked_fill(future, float("-inf"))
        x = x + self.out((scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d))
        h = self.fc1(self.norm2(x))
        return x + self.fc2(h * torch.sigmoid(1.702 * h))  # quick GELU


class TextEncoder(nn.Module):
    """Token ids (padded to 77) → one 768-wide vector per position: the keys and values of cross-attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.token_embedding = nn.Embedding(c.vocab, c.text_dim)
        self.position_embedding = nn.Embedding(c.max_tokens, c.text_dim)
        self.layers = nn.ModuleList(TextEncoderLayer(c) for _ in range(c.text_layers))
        self.final_norm = nn.LayerNorm(c.text_dim)

    def forward(self, ids):  # ids: (batch, 77)
        x = self.token_embedding(ids) + self.position_embedding.weight[: ids.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# %% [timestep-embedding] Timestep embedding
def sinusoid(t, dim):
    """cos then sin of t at dim/2 frequencies from 1 down to 1/10000."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half)
    angles = t[:, None].float() * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class TimestepEmbedding(nn.Module):
    """t → sinusoid (320) → Linear, SiLU, Linear → temb (1280), which every ResNet block reads."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.linear_1 = nn.Linear(width, 4 * width)
        self.linear_2 = nn.Linear(4 * width, 4 * width)

    def forward(self, t):
        return self.linear_2(F.silu(self.linear_1(sinusoid(t, self.width))))


# %% [resnet] ResNet block with the timestep
class ResnetBlock(nn.Module):
    """Two 3 × 3 convolutions with a skip around them; the timestep (if any) is added between them as one
    number per channel. The VAE uses the same block without the timestep."""

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


# %% [transformer] Transformer block: self-attention, cross-attention, feed-forward
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


class TransformerBlock(nn.Module):
    """Feature map → one token per pixel → self-attention, cross-attention to the text, GEGLU feed-forward
    (each after a LayerNorm, each added back) → feature map, added to the input."""

    def __init__(self, width: int, c: Config):
        super().__init__()
        self.norm = nn.GroupNorm(c.groups, width, eps=1e-6)
        self.proj_in = nn.Conv2d(width, width, 1)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.self_attention = Attention(width, c.heads)
        self.cross_attention = Attention(width, c.heads, context_dim=c.text_dim)
        self.ff_in = nn.Linear(width, 2 * 4 * width)  # GEGLU: a value half and a gate half
        self.ff_out = nn.Linear(4 * width, width)
        self.proj_out = nn.Conv2d(width, width, 1)

    def forward(self, x, text):  # x: (batch, width, h, w), text: (batch, 77, 768)
        b, width, h, w = x.shape
        tokens = self.proj_in(self.norm(x)).flatten(2).transpose(1, 2)  # (batch, h·w, width)
        tokens = tokens + self.self_attention(self.norm1(tokens))
        tokens = tokens + self.cross_attention(self.norm2(tokens), text)
        value, gate = self.ff_in(self.norm3(tokens)).chunk(2, dim=-1)
        tokens = tokens + self.ff_out(value * F.gelu(gate))
        return x + self.proj_out(tokens.transpose(1, 2).reshape(b, width, h, w))


# %% [unet] U-Net levels and skips
class Level(nn.Module):
    def __init__(self, resnets, attentions, resample=None):
        super().__init__()
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)  # transformer blocks (U-Net) or self-attention (VAE), or none
        self.resample = resample


class UNet(nn.Module):
    """Noisy latent, timestep and text → predicted noise. Down: each level halves the latent and saves
    every block's output. Up: each block concatenates the matching saved output to its input."""

    def __init__(self, c: Config):
        super().__init__()
        w, n, temb = c.widths, c.layers_per_level, 4 * c.widths[0]
        res = lambda c_in, c_out: ResnetBlock(c_in, c_out, temb, c.groups, c.eps)
        self.time_embedding = TimestepEmbedding(w[0])
        self.conv_in = nn.Conv2d(c.latent_channels, w[0], 3, padding=1)
        self.down = nn.ModuleList()
        for i, width in enumerate(w):
            text = i in c.text_levels
            down = nn.Conv2d(width, width, 3, stride=2, padding=1) if i < len(w) - 1 else None
            self.down.append(Level([res(w[max(i - 1, 0)] if j == 0 else width, width) for j in range(n)],
                                   [TransformerBlock(width, c) for _ in range(n)] if text else [], down))
        self.mid = Level([res(w[-1], w[-1]) for _ in range(2)], [TransformerBlock(w[-1], c)])
        skip_widths = [w[0]] + [width for i, width in enumerate(w) for _ in range(n + (i < len(w) - 1))]
        self.up = nn.ModuleList()
        width_in = w[-1]
        for i, width in enumerate(reversed(w)):
            level = len(w) - 1 - i
            resnets = [res((width_in if j == 0 else width) + skip_widths.pop(), width) for j in range(n + 1)]
            text = [TransformerBlock(width, c) for _ in range(n + 1)] if level in c.text_levels else []
            self.up.append(Level(resnets, text, nn.Conv2d(width, width, 3, padding=1) if level > 0 else None))
            width_in = width
        self.norm_out = nn.GroupNorm(c.groups, w[0], eps=c.eps)
        self.conv_out = nn.Conv2d(w[0], c.latent_channels, 3, padding=1)

    def forward(self, z, t, text):  # z: (batch, 4, h, w), t: (batch,), text: (batch, 77, 768)
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
                h = level.resample(h)  # stride 2: 32 → 16 → 8 → 4
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
    Its ResNet blocks have no timestep; a single self-attention sits at the bottom of each half."""

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
def guide(eps_empty, eps_prompt, scale=7.5):
    """Push the prediction further along what the prompt changes: ε̂ = ε(∅) + s · (ε(c) − ε(∅))."""
    return eps_empty + scale * (eps_prompt - eps_empty)


# %% [ddim-step] DDIM step
def noise_schedule(c: Config):
    """"Scaled linear": √β grows linearly, from √0.00085 to √0.012."""
    betas = torch.linspace(c.beta_start**0.5, c.beta_end**0.5, c.timesteps) ** 2
    return torch.cumprod(1 - betas, dim=0)  # ᾱ_t


def ddim_timesteps(steps, c: Config):
    """Every 1000/steps-th timestep, offset by one: 951, 901, …, 1 for 20 steps."""
    return torch.arange(0, steps).flip(0) * (c.timesteps // steps) + 1


def ddim_step(z, eps, t, alphas_cumprod, steps):
    """Deterministic (η = 0): estimate the clean latent, then re-noise it to the previous timestep
    with the same predicted noise."""
    a_t = alphas_cumprod[t]
    prev = t - len(alphas_cumprod) // steps
    a_prev = alphas_cumprod[prev] if prev >= 0 else alphas_cumprod[0]
    x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
    return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps, x0


# %% [euler-step] Euler step
def euler_sigmas(alphas_cumprod, steps):
    """Noise levels σ = √((1 − ᾱ) / ᾱ) at evenly spaced timesteps 999 … 0, then 0."""
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    timesteps = torch.linspace(0, len(sigmas) - 1, steps, dtype=torch.float64).flip(0)
    low = timesteps.floor().long()
    high = (low + 1).clamp(max=len(sigmas) - 1)
    frac = timesteps - low
    interpolated = (1 - frac) * sigmas[low].double() + frac * sigmas[high].double()
    return timesteps.float(), torch.cat([interpolated.float(), torch.zeros(1)])


def euler_step(z, eps, sigma, sigma_next):
    """Here z = x_0 + σ · ε (the U-Net reads z / √(σ² + 1)). One Euler step of dz/dσ = ε̂ from σ to σ_next."""
    x0 = z - sigma * eps
    return z + (sigma_next - sigma) * (z - x0) / sigma, x0


# %% [dpm-solver-step] DPM-Solver++ (2M) step
def dpm_x0(z, eps, sigma):
    """DPM-Solver++ works with x̂_0, not ε̂: z = α · x_0 + σ_t · ε, with α = 1/√(σ² + 1) and σ_t = σ · α."""
    alpha = 1 / (sigma**2 + 1) ** 0.5
    return (z - sigma * alpha * eps) / alpha


def dpm_solver_step(z, x0, x0_before, sigma, sigma_next, sigma_before=None):
    """Works in log signal-to-noise λ = log(α / σ_t), with α = 1/√(σ² + 1), σ_t = σ · α. First order (the
    DDIM step) when there is no earlier x̂_0 or σ_next = 0; otherwise the second-order multistep update that
    also uses the slope from the previous step's x̂_0."""
    alpha = lambda s: 1 / (s**2 + 1) ** 0.5
    lam = lambda s: torch.log(alpha(s)) - torch.log(s * alpha(s))
    if sigma_next == 0:
        return x0
    h = lam(sigma_next) - lam(sigma)
    a_next, s_next, s_now = alpha(sigma_next), sigma_next * alpha(sigma_next), sigma * alpha(sigma)
    z_next = (s_next / s_now) * z - a_next * (torch.exp(-h) - 1) * x0
    if x0_before is None:
        return z_next
    slope = (h / (lam(sigma) - lam(sigma_before))) * (x0 - x0_before)
    return z_next - 0.5 * a_next * (torch.exp(-h) - 1) * slope


# %% [model] Text to image
@torch.no_grad()
def sample(unet, vae, prompt, empty, c: Config, generator, size=256, steps=20, scale=7.5):
    """DDIM with classifier-free guidance: both passes in one batch, then the VAE decodes the final latent.
    \`prompt\` and \`empty\` are the text encoder's outputs, (1, 77, 768) each."""
    alphas_cumprod = noise_schedule(c)
    latent = size // 2 ** (len(c.vae_widths) - 1)  # 256 px → 32
    z = torch.randn(1, c.latent_channels, latent, latent, generator=generator)
    text = torch.cat([empty, prompt])
    for t in ddim_timesteps(steps, c):
        eps_empty, eps_prompt = unet(torch.cat([z, z]), t.repeat(2), text).chunk(2)
        z, _ = ddim_step(z, guide(eps_empty, eps_prompt, scale), t, alphas_cumprod, steps)
    return (vae.decode(z / c.vae_scale) / 2 + 0.5).clamp(0, 1)
`;export{e as default};
