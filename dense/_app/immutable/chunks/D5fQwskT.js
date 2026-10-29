const e=`# CogVideoX-2B, the text-to-video expert transformer in the CogVideoX tab, written as plain PyTorch blocks.
# Text and video tokens share one sequence: one attention over both, but each kind has its own adaLN
# ("expert"). It predicts v for a 16-channel latent of CogVideoX's 3-D causal VAE; DDIM turns v into steps.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 16
    patch: int = 2  # 2 × 2 latent positions of one frame per token
    d_model: int = 1920
    n_heads: int = 30  # 1920 / 30 = 64 dimensions per head
    n_layers: int = 30
    d_ffn: int = 7680  # 4 × d_model
    text_dim: int = 4096  # T5-XXL's width
    text_tokens: int = 226
    time_dim: int = 512
    spatial_scale: float = 1.875  # positions are divided by this before the sin-cos (trained at 60 × 90)
    temporal_scale: float = 1.0


# %% [vae] 3-D causal VAE
class CausalConv3d(nn.Conv3d):
    """The VAE's 3-D convolution. Before frame 0 it repeats frame 0 instead of looking ahead, so no frame
    sees a later one; height and width are zero-padded as usual."""

    def __init__(self, c_in, c_out, kernel):
        super().__init__(c_in, c_out, kernel, padding=(0, kernel[1] // 2, kernel[2] // 2))
        self.past = kernel[0] - 1

    def forward(self, x):  # x: (batch, channels, frames, height, width)
        return super().forward(torch.cat([x[:, :, :1]] * self.past + [x], dim=2))


def latent_shape(frames, height, width, channels=16):
    """Frame 0, then every 4 frames, become one latent frame; 8 × smaller in height and width.
    9 frames of 256 × 384 × 3 become 3 × 32 × 48 × 16 numbers."""
    return (1 + (frames - 1) // 4, channels, height // 8, width // 8)


# %% [patchify] Patch embedding and the joint sequence
class PatchEmbedding(nn.Module):
    """Each frame is cut into 2 × 2 patches (64 numbers) and projected to d_model; the 226 T5 vectors are
    projected to d_model too. The transformer sees one sequence: 226 text tokens, then the video tokens."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.proj = nn.Conv2d(c.latent_channels, c.d_model, kernel_size=c.patch, stride=c.patch)
        self.text_proj = nn.Linear(c.text_dim, c.d_model)

    def forward(self, text, latent):  # text: (batch, 226, 4096); latent: (batch, F, 16, H, W)
        b, f, ch, h, w = latent.shape
        video = self.proj(latent.reshape(b * f, ch, h, w))  # (b · F, d_model, H/2, W/2)
        video = video.view(b, f, self.c.d_model, -1).transpose(2, 3).reshape(b, -1, self.c.d_model)
        return torch.cat([self.text_proj(text), video], dim=1)  # (batch, 226 + F · H/2 · W/2, d_model)


# %% [position] 3-D sin-cos position embedding
def sincos(d, positions):
    """d/2 sines then d/2 cosines of each position at frequencies 10000^(−2i/d)."""
    frequencies = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2))
    angles = positions.double()[:, None] * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=1)


def position_embedding(c: Config, frames, rows, cols):
    """Added to each video token, not learned: a quarter of d_model encodes the latent frame, the rest the
    column then the row. Text tokens get zeros. Positions are divided by 1.875, so this 16 × 24 grid spans
    the same range as 8.5 × 12.8 patches of the 30 × 45 grid the model was trained on."""
    temporal_dim, spatial_dim = c.d_model // 4, 3 * c.d_model // 4
    f, r, k = torch.meshgrid(torch.arange(frames), torch.arange(rows), torch.arange(cols), indexing="ij")
    f, r, k = f.flatten(), r.flatten(), k.flatten()
    video = torch.cat([
        sincos(temporal_dim, f / c.temporal_scale),
        sincos(spatial_dim // 2, k / c.spatial_scale),  # column
        sincos(spatial_dim // 2, r / c.spatial_scale),  # row
    ], dim=1)
    return torch.cat([torch.zeros(c.text_tokens, c.d_model, dtype=video.dtype), video]).float()


# %% [conditioning] Timestep embedding
class TimestepEmbedding(nn.Module):
    """t → cos/sin at d_model / 2 frequencies → Linear, SiLU, Linear → temb (512). Every block reads it."""

    def __init__(self, c: Config):
        super().__init__()
        self.d_model = c.d_model
        self.linear_1 = nn.Linear(c.d_model, c.time_dim)
        self.linear_2 = nn.Linear(c.time_dim, c.time_dim)

    def sinusoid(self, t):  # t: (batch,) integer timesteps 0…999
        half = self.d_model // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        angles = t[:, None].float() * frequencies
        return torch.cat([angles.cos(), angles.sin()], dim=1)

    def forward(self, t):
        return self.linear_2(F.silu(self.linear_1(self.sinusoid(t))))  # (batch, 512)


# %% [expert-adaln] Expert adaptive LayerNorm
class ExpertAdaLN(nn.Module):
    """One LayerNorm, two sets of shift, scale and gate from temb: the vision expert's for video tokens and
    the text expert's for text tokens. Text and video live in different spaces, so each gets its own."""

    def __init__(self, c: Config):
        super().__init__()
        self.linear = nn.Linear(c.time_dim, 6 * c.d_model)
        self.norm = nn.LayerNorm(c.d_model, eps=1e-5)

    def forward(self, video, text, temb):
        shift, scale, gate, text_shift, text_scale, text_gate = self.linear(F.silu(temb))[:, None].chunk(6, dim=-1)
        video = self.norm(video) * (1 + scale) + shift
        text = self.norm(text) * (1 + text_scale) + text_shift
        return video, text, gate, text_gate


# %% [attention] Joint text–video attention
class JointAttention(nn.Module):
    """Plain self-attention over [text; video]: every video token reads the prompt and every frame, and
    every text token reads the video. No mask, so the prompt's padding tokens are keys too. Each head's
    q and k pass a LayerNorm; positions came in through the sin-cos embedding, so there is no RoPE."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.norm_q = nn.LayerNorm(head_dim, eps=1e-6)
        self.norm_k = nn.LayerNorm(head_dim, eps=1e-6)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, video, text):
        x = torch.cat([text, video], dim=1)  # (batch, 226 + video tokens, d_model)
        b, n, d = x.shape
        q = self.norm_q(self.q(x).view(b, n, self.n_heads, -1).transpose(1, 2))
        k = self.norm_k(self.k(x).view(b, n, self.n_heads, -1).transpose(1, 2))
        v = self.v(x).view(b, n, self.n_heads, -1).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (b, heads, n, n)
        out = self.out((weights @ v).transpose(1, 2).reshape(b, n, d))
        return out[:, text.shape[1]:], out[:, : text.shape[1]]  # video, text


# %% [mlp] Feed-forward
class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_ffn)
        self.down = nn.Linear(c.d_ffn, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] Expert transformer block
class Block(nn.Module):
    """Expert adaLN, joint attention, gated residual; expert adaLN, one feed-forward over both, gated
    residual. Text and video keep separate residual streams but meet in every attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = ExpertAdaLN(c)
        self.attention = JointAttention(c)
        self.norm2 = ExpertAdaLN(c)
        self.mlp = FeedForward(c)

    def forward(self, video, text, temb):
        v, t, gate, text_gate = self.norm1(video, text, temb)
        v, t = self.attention(v, t)
        video, text = video + gate * v, text + text_gate * t
        v, t, gate, text_gate = self.norm2(video, text, temb)
        out = self.mlp(torch.cat([t, v], dim=1))
        return video + gate * out[:, text.shape[1]:], text + text_gate * out[:, : text.shape[1]]


# %% [unpatchify] Output and unpatchify
class Output(nn.Module):
    """LayerNorm, then adaLN from temb (shift and scale), a Linear to 2 × 2 × 16 numbers per video token,
    and the tokens back into a (frames, 16, H, W) v-prediction. The text stream is dropped."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.norm_final = nn.LayerNorm(c.d_model, eps=1e-5)
        self.ada = nn.Linear(c.time_dim, 2 * c.d_model)
        self.norm = nn.LayerNorm(c.d_model, eps=1e-5)
        self.proj = nn.Linear(c.d_model, c.patch * c.patch * c.latent_channels)

    def forward(self, video, temb, frames, rows, cols):
        shift, scale = self.ada(F.silu(temb))[:, None].chunk(2, dim=-1)
        x = self.proj(self.norm(self.norm_final(video)) * (1 + scale) + shift)
        p = self.c.patch
        x = x.reshape(-1, frames, rows, cols, self.c.latent_channels, p, p)
        return x.permute(0, 1, 4, 2, 5, 3, 6).reshape(-1, frames, self.c.latent_channels, rows * p, cols * p)


# %% [ddim-v] v-prediction with DDIM
def alphas_cumprod(steps=1000, beta_start=0.00085, beta_end=0.012, snr_shift=3.0):
    """ᾱ_t, the signal kept at step t: the scaled-linear schedule, its SNR divided by 3 (as SD3 does for
    large images), then rescaled so the last step is pure noise (ᾱ₉₉₉ = 0)."""
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, steps, dtype=torch.float64) ** 2
    a = torch.cumprod(1 - betas, dim=0)
    a = a / (snr_shift + (1 - snr_shift) * a)
    root = a.sqrt()
    root = (root - root[-1]) * root[0] / (root[0] - root[-1])
    return root**2


def timesteps(steps, train_steps=1000):
    """"Trailing" spacing: 999, 979, …, 19 for 50 steps, so the loop starts from pure noise."""
    return torch.round(torch.arange(train_steps, 0, -train_steps / steps)).long() - 1


def ddim_v_step(x, v, a_t, a_prev):
    """The model predicts v = √ᾱ ε − √(1 − ᾱ) x₀. Undo it for x₀, then take DDIM's deterministic step to the
    previous noise level: the same ε, less of it."""
    x0 = a_t**0.5 * x - (1 - a_t) ** 0.5 * v
    noise_share = ((1 - a_prev) / (1 - a_t)) ** 0.5
    return noise_share * x + (a_prev**0.5 - a_t**0.5 * noise_share) * x0, x0


def guided(v_text, v_negative, scale=6.0):
    return v_negative + scale * (v_text - v_negative)


# %% [model] CogVideoX transformer
class CogVideoX(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.time = TimestepEmbedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)

    def forward(self, latent, t, text):  # latent: (batch, F, 16, H, W); t: (batch,); text: (batch, 226, 4096)
        frames, rows, cols = latent.shape[1], latent.shape[3] // self.c.patch, latent.shape[4] // self.c.patch
        x = self.patchify(text, latent) + position_embedding(self.c, frames, rows, cols)
        text, video = x[:, : self.c.text_tokens], x[:, self.c.text_tokens:]
        temb = self.time(t)
        for block in self.blocks:
            video, text = block(video, text, temb)
        return self.output(video, temb, frames, rows, cols)  # v, same shape as the latent
`;export{e as default};
