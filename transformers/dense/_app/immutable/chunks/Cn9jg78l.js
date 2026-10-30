const e=`# AudioLDM (cvssp/audioldm-s-full-v2), the text-to-audio latent U-Net of the AudioLDM tab, written as plain PyTorch
# blocks. CLAP's text tower turns the prompt into one unit-length 512-vector; the U-Net reads it only through its class
# embedding, side by side with the timestep's, so no layer attends to the text. The U-Net denoises an 8-channel latent
# of a log-mel spectrogram (1/4 of its 100 frames a second, 1/4 of its 64 mel bins); the VAE decodes the latent to the
# spectrogram and HiFi-GAN turns that into 16 kHz audio, 160 samples per mel frame.
# Each "# %% [id]" section is one step of the tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 8
    widths: tuple = (128, 256, 384, 640)  # U-Net channels at latent sizes 1, 1/2, 1/4, 1/8
    attention_levels: tuple = (1, 2, 3)  # levels with a transformer block after each ResNet block
    layers_per_level: int = 2
    heads: int = 8  # head width = channels / 8: 32, 48, 80
    groups: int = 32
    eps: float = 1e-5
    clap_dim: int = 512  # CLAP's joint text–audio space
    text_dim: int = 768  # RoBERTa-base
    vocab: int = 50265
    text_layers: int = 12
    text_heads: int = 12
    text_mlp: int = 3072
    max_positions: int = 514
    pad_id: int = 1  # <pad>; positions count from pad_id + 1
    vae_widths: tuple = (128, 256, 512)  # the VAE's channels at 1, 1/2, 1/4 of the spectrogram
    vae_scale: float = 0.9392935633659363  # latents × 0.939 have about unit variance
    mel_bins: int = 64
    sample_rate: int = 16000
    upsample_rates: tuple = (5, 4, 2, 2, 2)  # × 160 samples: one mel frame per 10 ms
    upsample_kernels: tuple = (16, 16, 8, 4, 4)
    vocoder_width: int = 1024
    resblock_kernels: tuple = (3, 7, 11)
    resblock_dilations: tuple = (1, 3, 5)
    timesteps: int = 1000
    beta_start: float = 0.0015
    beta_end: float = 0.0195


# %% [text-encoder] CLAP text encoder
class TextLayer(nn.Module):
    """RoBERTa layer (post-LayerNorm): attention over the unpadded tokens, add, LayerNorm; GELU MLP, add, LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        d, self.heads = c.text_dim, c.text_heads
        self.q, self.k, self.v, self.attn_out = (nn.Linear(d, d) for _ in range(4))
        self.norm1 = nn.LayerNorm(d, eps=1e-12)
        self.fc1, self.fc2 = nn.Linear(d, c.text_mlp), nn.Linear(c.text_mlp, d)
        self.norm2 = nn.LayerNorm(d, eps=1e-12)

    def forward(self, x, keep):  # x: (batch, n, 768), keep: (batch, n) True for real tokens
        b, n, d = x.shape
        q, k, v = (proj(x).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        scores = (q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)).masked_fill(~keep[:, None, None], -torch.inf)
        x = self.norm1(x + self.attn_out((scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, d)))
        return self.norm2(x + self.fc2(F.gelu(self.fc1(x))))


class TextEncoder(nn.Module):
    """Token ids (padded to 512) → RoBERTa → the first token, through tanh(Linear) and a ReLU MLP → one 512-vector,
    scaled to length 1. That single vector is all the U-Net learns about the prompt."""

    def __init__(self, c: Config):
        super().__init__()
        self.pad_id = c.pad_id
        self.token_embedding = nn.Embedding(c.vocab, c.text_dim)
        self.position_embedding = nn.Embedding(c.max_positions, c.text_dim)
        self.type_embedding = nn.Parameter(torch.zeros(1, c.text_dim))  # the one token type's vector
        self.embedding_norm = nn.LayerNorm(c.text_dim, eps=1e-12)
        self.layers = nn.ModuleList(TextLayer(c) for _ in range(c.text_layers))
        self.pooler = nn.Linear(c.text_dim, c.text_dim)
        self.projection_1 = nn.Linear(c.text_dim, c.clap_dim)
        self.projection_2 = nn.Linear(c.clap_dim, c.clap_dim)

    def forward(self, ids, keep):  # ids, keep: (batch, n)
        positions = keep.long().cumsum(1) * keep + self.pad_id  # 2, 3, … for tokens; padding sits at pad_id
        x = self.token_embedding(ids) + self.type_embedding + self.position_embedding(positions)
        x = self.embedding_norm(x)
        for layer in self.layers:
            x = layer(x, keep)
        pooled = torch.tanh(self.pooler(x[:, 0]))  # <s>, the first token
        embedding = self.projection_2(F.relu(self.projection_1(pooled)))
        return F.normalize(embedding, dim=-1)  # (batch, 512), unit length


# %% [timestep-embedding] Timestep embedding
def sinusoid(t, dim):
    """cos then sin of t at dim/2 frequencies from 1 down to 1/10000."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half) / half)
    angles = t[:, None].float() * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class TimestepEmbedding(nn.Module):
    """t → sinusoid (128) → Linear, SiLU, Linear → 512."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.linear_1 = nn.Linear(width, 4 * width)
        self.linear_2 = nn.Linear(4 * width, 4 * width)

    def forward(self, t):
        return self.linear_2(F.silu(self.linear_1(sinusoid(t, self.width))))


# %% [class-embedding] The text as a class embedding
class Conditioning(nn.Module):
    """The CLAP vector goes through one Linear (diffusers' "simple projection" class embedding) and sits next to the
    timestep embedding: [temb ‖ text] (1,024) is the one vector every ResNet block reads. Classifier-free guidance's
    empty prompt is CLAP's vector for "", not zeros."""

    def __init__(self, c: Config):
        super().__init__()
        self.time_embedding = TimestepEmbedding(c.widths[0])
        self.class_embedding = nn.Linear(c.clap_dim, 4 * c.widths[0])

    def forward(self, t, text):  # t: (batch,), text: (batch, 512)
        return torch.cat([self.time_embedding(t), self.class_embedding(text)], dim=-1)  # (batch, 1024)


# %% [resnet] ResNet block with the timestep and the text
class ResnetBlock(nn.Module):
    """Two 3 × 3 convolutions with a skip around them; [temb ‖ text] is added between them as one number per channel.
    The VAE uses the same block with nothing added."""

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


# %% [transformer] Transformer block: two self-attentions and a feed-forward
class Attention(nn.Module):
    """Multi-head self-attention over the latent's pixels (time × frequency)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim, dim, bias=False) for _ in range(3))
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        split = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = split(self.to_q(x)), split(self.to_k(x)), split(self.to_v(x))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)
        return self.to_out((weights @ v).transpose(1, 2).reshape(b, n, d))


class TransformerBlock(nn.Module):
    """Feature map → one token per latent pixel → self-attention, then a second self-attention where SD would
    cross-attend to the text (the pipeline passes no text tokens), then a GEGLU feed-forward, each after a LayerNorm
    and added back → feature map, added to the input."""

    def __init__(self, width: int, c: Config):
        super().__init__()
        self.norm = nn.GroupNorm(c.groups, width, eps=1e-6)
        self.proj_in = nn.Conv2d(width, width, 1)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention_1 = Attention(width, c.heads)
        self.attention_2 = Attention(width, c.heads)
        self.ff_in = nn.Linear(width, 2 * 4 * width)  # GEGLU: a value half and a gate half
        self.ff_out = nn.Linear(4 * width, width)
        self.proj_out = nn.Conv2d(width, width, 1)

    def forward(self, x):  # x: (batch, width, h, w)
        b, width, h, w = x.shape
        tokens = self.proj_in(self.norm(x)).flatten(2).transpose(1, 2)  # (batch, h·w, width)
        tokens = tokens + self.attention_1(self.norm1(tokens))
        tokens = tokens + self.attention_2(self.norm2(tokens))
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
    """Noisy latent, timestep and CLAP vector → predicted noise. Down: each level halves both axes and saves every
    block's output. Up: each block concatenates the matching saved output to its input."""

    def __init__(self, c: Config):
        super().__init__()
        w, n, temb = c.widths, c.layers_per_level, 2 * 4 * c.widths[0]  # [temb ‖ text]: 1,024
        res = lambda c_in, c_out: ResnetBlock(c_in, c_out, temb, c.groups, c.eps)
        self.conditioning = Conditioning(c)
        self.conv_in = nn.Conv2d(c.latent_channels, w[0], 3, padding=1)
        self.down = nn.ModuleList()
        for i, width in enumerate(w):
            attend = i in c.attention_levels
            down = nn.Conv2d(width, width, 3, stride=2, padding=1) if i < len(w) - 1 else None
            self.down.append(Level([res(w[max(i - 1, 0)] if j == 0 else width, width) for j in range(n)],
                                   [TransformerBlock(width, c) for _ in range(n)] if attend else [], down))
        self.mid = Level([res(w[-1], w[-1]) for _ in range(2)], [TransformerBlock(w[-1], c)])
        skip_widths = [w[0]] + [width for i, width in enumerate(w) for _ in range(n + (i < len(w) - 1))]
        self.up = nn.ModuleList()
        width_in = w[-1]
        for i, width in enumerate(reversed(w)):
            level = len(w) - 1 - i
            resnets = [res((width_in if j == 0 else width) + skip_widths.pop(), width) for j in range(n + 1)]
            attend = [TransformerBlock(width, c) for _ in range(n + 1)] if level in c.attention_levels else []
            self.up.append(Level(resnets, attend, nn.Conv2d(width, width, 3, padding=1) if level > 0 else None))
            width_in = width
        self.norm_out = nn.GroupNorm(c.groups, w[0], eps=c.eps)
        self.conv_out = nn.Conv2d(w[0], c.latent_channels, 3, padding=1)

    def forward(self, z, t, text):  # z: (batch, 8, frames/4, 16), t: (batch,), text: (batch, 512)
        temb = self.conditioning(t, text)
        h = self.conv_in(z)
        skips = [h]
        for level in self.down:
            for j, resnet in enumerate(level.resnets):
                h = resnet(h, temb)
                if len(level.attentions):
                    h = level.attentions[j](h)
                skips.append(h)
            if level.resample is not None:
                h = level.resample(h)  # stride 2 on both axes: 16 mel columns → 8 → 4 → 2
                skips.append(h)
        h = self.mid.resnets[1](self.mid.attentions[0](self.mid.resnets[0](h, temb)), temb)
        for level in self.up:
            for j, resnet in enumerate(level.resnets):
                h = resnet(torch.cat([h, skips.pop()], dim=1), temb)
                if len(level.attentions):
                    h = level.attentions[j](h)
            if level.resample is not None:
                h = level.resample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.conv_out(F.silu(self.norm_out(h)))  # ε̂: (batch, 8, frames/4, 16)


# %% [guidance] Classifier-free guidance
def guide(eps_empty, eps_prompt, scale=2.5):
    """Push the prediction further along what the prompt changes: ε̂ = ε(∅) + s · (ε(c) − ε(∅))."""
    return eps_empty + scale * (eps_prompt - eps_empty)


# %% [ddim-step] DDIM step
def noise_schedule(c: Config):
    """"Scaled linear": √β grows linearly, from √0.0015 to √0.0195 (noisier than SD's 0.00085 to 0.012)."""
    betas = torch.linspace(c.beta_start**0.5, c.beta_end**0.5, c.timesteps) ** 2
    return torch.cumprod(1 - betas, dim=0)  # ᾱ_t


def ddim_timesteps(steps, c: Config):
    """Every 1000/steps-th timestep, offset by one: 951, 901, …, 1 for 20 steps."""
    return torch.arange(0, steps).flip(0) * (c.timesteps // steps) + 1


def ddim_step(z, eps, t, alphas_cumprod, steps):
    """Deterministic (η = 0): estimate the clean latent, then re-noise it to the previous timestep with the same
    predicted noise."""
    a_t = alphas_cumprod[t]
    prev = t - len(alphas_cumprod) // steps
    a_prev = alphas_cumprod[prev] if prev >= 0 else alphas_cumprod[0]
    x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
    return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps, x0


# %% [vae] VAE: log-mel spectrogram ↔ latent
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
    """Encoder: a 1 × frames × 64 log-mel spectrogram → mean and log-variance of an 8 × frames/4 × 16 latent. Decoder:
    the mirror image. The spectrogram is treated as a one-channel image: time down, mel bins across."""

    def __init__(self, c: Config):
        super().__init__()
        w, g, n = c.vae_widths, c.groups, c.layers_per_level
        res = lambda c_in, c_out: ResnetBlock(c_in, c_out, None, g, 1e-6)
        self.enc_in = nn.Conv2d(1, w[0], 3, padding=1)
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
        self.dec_out = nn.Conv2d(w[0], 1, 3, padding=1)

    @staticmethod
    def middle(level, h):
        return level.resnets[1](level.attentions[0](level.resnets[0](h)))

    def encode(self, mel):  # mel: (batch, 1, frames, 64) → mean, log-variance (batch, 8, frames/4, 16)
        h = self.enc_in(mel)
        for level in self.enc_down:
            for resnet in level.resnets:
                h = resnet(h)
            if level.resample is not None:
                h = level.resample(F.pad(h, (0, 1, 0, 1)))
        h = self.enc_out(F.silu(self.enc_norm(self.middle(self.enc_mid, h))))
        mean, logvar = self.quant(h).chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)

    def decode(self, z):  # z: (batch, 8, frames/4, 16), unscaled → (batch, 1, frames, 64) log-mel
        h = self.middle(self.dec_mid, self.dec_in(self.post_quant(z)))
        for level in self.dec_up:
            for resnet in level.resnets:
                h = resnet(h)
            if level.resample is not None:
                h = level.resample(F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return self.dec_out(F.silu(self.dec_norm(h)))


# %% [vocoder] HiFi-GAN vocoder: spectrogram → waveform
class ResidualStack(nn.Module):
    """Three dilated convolutions of one kernel size, each followed by a plain one and added back: each stack hears
    a different span of the waveform."""

    def __init__(self, channels, kernel, dilations, slope=0.1):
        super().__init__()
        self.slope = slope
        self.convs1 = nn.ModuleList(nn.Conv1d(channels, channels, kernel, dilation=d, padding=(kernel * d - d) // 2)
                                    for d in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)
                                    for _ in dilations)

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x = x + conv2(F.leaky_relu(conv1(F.leaky_relu(x, self.slope)), self.slope))
        return x


class Vocoder(nn.Module):
    """64 mel bins per frame → 1,024 channels → five transposed convolutions (× 5, 4, 2, 2, 2 = 160 samples per frame,
    halving the channels each time), each followed by the average of three residual stacks (kernels 3, 7, 11) →
    one channel → tanh: audio in [−1, 1] at 16 kHz."""

    def __init__(self, c: Config):
        super().__init__()
        width, k = c.vocoder_width, len(c.resblock_kernels)
        self.kernels = k
        self.conv_pre = nn.Conv1d(c.mel_bins, width, 7, padding=3)
        self.upsampler = nn.ModuleList(
            nn.ConvTranspose1d(width // 2**i, width // 2 ** (i + 1), kernel, stride=rate, padding=(kernel - rate) // 2)
            for i, (rate, kernel) in enumerate(zip(c.upsample_rates, c.upsample_kernels)))
        self.resblocks = nn.ModuleList(
            ResidualStack(width // 2 ** (i + 1), kernel, c.resblock_dilations)
            for i in range(len(c.upsample_rates)) for kernel in c.resblock_kernels)
        self.conv_post = nn.Conv1d(width // 2 ** len(c.upsample_rates), 1, 7, padding=3)

    def forward(self, mel):  # mel: (batch, frames, 64) → (batch, frames · 160)
        h = self.conv_pre(mel.transpose(1, 2))
        for i, upsample in enumerate(self.upsampler):
            h = upsample(F.leaky_relu(h, 0.1))
            h = sum(block(h) for block in self.resblocks[i * self.kernels:(i + 1) * self.kernels]) / self.kernels
        return torch.tanh(self.conv_post(F.leaky_relu(h))).squeeze(1)


# %% [model] Text to audio
@torch.no_grad()
def sample(unet, vae, vocoder, prompt, empty, c: Config, generator, seconds=5.12, steps=20, scale=2.5):
    """DDIM with classifier-free guidance on the mel latent; the VAE decodes the final latent to a log-mel
    spectrogram and HiFi-GAN to audio. \`prompt\` and \`empty\` are the CLAP vectors of the prompt and of "", (1, 512)."""
    hop = math.prod(c.upsample_rates)
    frames = round(seconds * c.sample_rate / hop)  # 512 mel frames for 5.12 s
    alphas_cumprod = noise_schedule(c)
    shrink = 2 ** (len(c.vae_widths) - 1)
    z = torch.randn(1, c.latent_channels, frames // shrink, c.mel_bins // shrink, generator=generator)
    text = torch.cat([empty, prompt])
    for t in ddim_timesteps(steps, c):
        eps_empty, eps_prompt = unet(torch.cat([z, z]), t.repeat(2), text).chunk(2)
        z, _ = ddim_step(z, guide(eps_empty, eps_prompt, scale), t, alphas_cumprod, steps)
    mel = vae.decode(z / c.vae_scale)  # (1, 1, frames, 64)
    return vocoder(mel[:, 0])[:, : int(seconds * c.sample_rate)]
`;export{e as default};
