const e=`# LTX-Video 2B (v0.9.1), the text-to-video transformer in the LTX tab, written as plain PyTorch blocks.
# Its VAE compresses 1:192 (32 × 32 pixels × 8 frames × 3 colours → 128 numbers), so the transformer's patches
# are single latent positions: patchify happens in the VAE. T5-XXL encodes the prompt. The transformer predicts a
# flow-matching velocity; the VAE decoder, conditioned on a small timestep, takes the last denoising step.
# Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 128
    d_model: int = 2048
    n_heads: int = 32  # 2048 / 32 = 64 dimensions per head
    n_layers: int = 28
    d_ffn: int = 8192  # 4 × d_model
    text_dim: int = 4096  # T5-XXL's width
    frequency_dim: int = 256
    vae_patch: int = 4  # the VAE's first step: 4 × 4 pixels side by side become 48 channels
    vae_spatial: int = 32
    vae_temporal: int = 8
    rope_base: tuple = (20, 2048, 2048)  # frames, height, width the RoPE coordinates are divided by
    rope_theta: float = 10000.0


# %% [vae-patchify] Patchify in the VAE
def vae_patchify(video, p=4):
    """Space-to-depth, the LTX VAE's first operation: each 4 × 4 block of pixels becomes one position with
    3 · 4 · 4 = 48 channels, in the encoder's own channel order. The encoder then shrinks 8 × more in space
    and 8 × in time; a latent position covers 8 frames × 32 × 32 pixels."""
    b, c, f, h, w = video.shape
    x = video.reshape(b, c, f, 1, h // p, p, w // p, p)
    return x.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)  # (batch, 48, F, H/4, W/4)


def latent_shape(frames, height, width, channels=128):
    """Frame 0, then every 8 frames, become one latent frame; 32 × smaller in height and width.
    17 frames of 256 × 384 × 3 become 3 × 8 × 12 × 128 numbers, and each is one transformer token."""
    return (channels, 1 + (frames - 1) // 8, height // 32, width // 32)


def compression(c: Config):
    return 3 * c.vae_temporal * c.vae_spatial**2 / c.latent_channels  # 192


# %% [vae-decoder] The decoder's last denoising step
def timestep_sinusoid(t, dim=256):
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    angles = t[:, None].float() * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class DecoderHead(nn.Module):
    """The VAE decoder is conditioned on a timestep. The pipeline adds a little noise to the finished latent,
    z' = (1 − s) z + s ε with s = 0.025, and decodes at t = 0.03: the decoder removes that noise as it
    upsamples. This is its last stage: RMSNorm, a shift and scale from t, SiLU, a 3-D convolution, and
    depth-to-space back to 4 × 4 pixels per position."""

    def __init__(self, channels, out_channels=3, p=4):
        super().__init__()
        self.p = p
        self.time_in = nn.Linear(256, 2 * channels)
        self.time_out = nn.Linear(2 * channels, 2 * channels)
        self.table = nn.Parameter(torch.zeros(2, channels))
        self.conv_out = nn.Conv3d(channels, out_channels * p * p, 3, padding=(0, 1, 1))
        self.multiplier = nn.Parameter(torch.tensor(1000.0))  # learned; t = 0.03 enters as 30

    def forward(self, h, t):  # h: (batch, channels, F, H, W), the up-blocks' output; t: (batch,)
        h = h * torch.rsqrt(h.pow(2).mean(dim=1, keepdim=True) + 1e-8)  # RMSNorm over channels
        temb = self.time_out(F.silu(self.time_in(timestep_sinusoid(t * self.multiplier))))
        shift, scale = (temb.view(-1, 2, h.shape[1]) + self.table)[..., None, None, None].unbind(1)
        h = F.silu(h * (1 + scale) + shift)
        h = torch.cat([h[:, :, :1], h, h[:, :, -1:]], dim=2)  # not causal: repeat the edge frames
        x = self.conv_out(h)  # (batch, 3 · 4 · 4, F, H, W)
        b, _, f, hh, ww = x.shape
        x = x.reshape(b, -1, 1, self.p, self.p, f, hh, ww).permute(0, 1, 5, 2, 6, 4, 7, 3)
        return x.reshape(b, -1, f, hh * self.p, ww * self.p)  # depth-to-space


def decode_noise(z, noise, scale=0.025):
    return (1 - scale) * z + scale * noise


# %% [rope] 3-D rotary position embedding
def rope(c: Config, grid, scale):
    """cos and sin for every (token, channel). LTX rotates the whole 2048-wide q and k, before splitting
    heads. A token's frame, row and column are scaled to [0, 1] by \`scale\` (8 / 25 frames, 32 pixels per
    position) and the base sizes, then mapped to 2x − 1; frequencies run from 1 to θ, times π/2. The three
    axes are interleaved per frequency, and the 2 leftover channels are not rotated."""
    frames, rows, cols = grid
    f, r, k = torch.meshgrid(torch.arange(frames), torch.arange(rows), torch.arange(cols), indexing="ij")
    coords = torch.stack([f, r, k], dim=-1).reshape(-1, 3).float()
    coords = coords * torch.tensor(scale) / torch.tensor(c.rope_base, dtype=torch.float32)
    n = c.d_model // 6
    frequencies = c.rope_theta ** torch.linspace(0.0, 1.0, n) * math.pi / 2  # (341,)
    angles = (frequencies[None, None] * (coords[..., None] * 2 - 1)).transpose(-1, -2).flatten(1)  # (tokens, 1023)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    pad = c.d_model % 6
    cos = torch.cat([torch.ones(cos.shape[0], pad), cos], dim=-1)
    sin = torch.cat([torch.zeros(sin.shape[0], pad), sin], dim=-1)
    return cos, sin  # (tokens, d_model)


def rotate(x, cos, sin):  # x: (batch, tokens, d_model)
    real, imag = x.unflatten(-1, (-1, 2)).unbind(-1)
    turned = torch.stack([-imag, real], dim=-1).flatten(-2)
    return x * cos + turned * sin


# %% [conditioning] Timestep and caption conditioning
class Conditioning(nn.Module):
    """t → cos/sin → MLP → e (2048), and e → SiLU → Linear → 6 × 2048 shared by all blocks (adaLN-single,
    from PixArt). The T5 vectors go through a two-layer MLP to 2048 for cross-attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.time_in = nn.Linear(c.frequency_dim, c.d_model)
        self.time_out = nn.Linear(c.d_model, c.d_model)
        self.modulation = nn.Linear(c.d_model, 6 * c.d_model)
        self.text_in = nn.Linear(c.text_dim, c.d_model)
        self.text_out = nn.Linear(c.d_model, c.d_model)

    def forward(self, t, text):
        e = self.time_out(F.silu(self.time_in(timestep_sinusoid(t))))
        text = self.text_out(F.gelu(self.text_in(text), approximate="tanh"))
        return e, self.modulation(F.silu(e)), text  # (b, d), (b, 6d), (b, L, d)


# %% [attention] Full 3-D self-attention
class SelfAttention(nn.Module):
    """Every latent position attends to every other one, across frames. q and k are RMS-normalised across
    all heads and rotated by the 3-D RoPE."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.norm_q = nn.RMSNorm(c.d_model, eps=1e-5)
        self.norm_k = nn.RMSNorm(c.d_model, eps=1e-5)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x, cos, sin):
        b, n, d = x.shape
        q = rotate(self.norm_q(self.q(x)), cos, sin).view(b, n, self.n_heads, -1).transpose(1, 2)
        k = rotate(self.norm_k(self.k(x)), cos, sin).view(b, n, self.n_heads, -1).transpose(1, 2)
        v = self.v(x).view(b, n, self.n_heads, -1).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d))


# %% [cross-attention] Cross-attention to T5
class CrossAttention(nn.Module):
    """Video positions ask, the prompt's T5 tokens answer. The residual stream goes in without a norm; the
    prompt's padding is masked out (−10000 before the softmax). No RoPE."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.norm_q = nn.RMSNorm(c.d_model, eps=1e-5)
        self.norm_k = nn.RMSNorm(c.d_model, eps=1e-5)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x, text, mask):  # mask: (batch, L), 1 for the prompt's tokens, 0 for padding
        b, n, d = x.shape
        q = self.norm_q(self.q(x)).view(b, n, self.n_heads, -1).transpose(1, 2)
        k = self.norm_k(self.k(text)).view(b, -1, self.n_heads, q.shape[-1]).transpose(1, 2)
        v = self.v(text).view(b, -1, self.n_heads, q.shape[-1]).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) + (1 - mask[:, None, None].float()) * -10000.0
        return self.out((scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, d))


# %% [mlp] Feed-forward
class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_ffn)
        self.down = nn.Linear(c.d_ffn, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] LTX block
class Block(nn.Module):
    """RMSNorm + adaLN, self-attention, gated residual; cross-attention, plain residual; RMSNorm + adaLN,
    feed-forward, gated residual. The six vectors are the shared modulation plus this block's own table."""

    def __init__(self, c: Config):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(6, c.d_model))
        self.norm = nn.RMSNorm(c.d_model, eps=1e-6, elementwise_affine=False)
        self.attention = SelfAttention(c)
        self.cross = CrossAttention(c)
        self.mlp = FeedForward(c)

    def forward(self, x, text, mask, modulation, cos, sin):  # modulation: (batch, 6 · d_model)
        shift1, scale1, gate1, shift2, scale2, gate2 = (self.table + modulation.view(-1, 1, 6, x.shape[-1])).unbind(2)
        x = x + gate1 * self.attention(self.norm(x) * (1 + scale1) + shift1, cos, sin)
        x = x + self.cross(x, text, mask)
        return x + gate2 * self.mlp(self.norm(x) * (1 + scale2) + shift2)


# %% [unpatchify] Output
class Output(nn.Module):
    """LayerNorm, shift and scale from e, a Linear back to the 128 latent channels. Each token is one latent
    position, so unpatchify is only a reshape to (128, F, H, W)."""

    def __init__(self, c: Config):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(2, c.d_model))
        self.norm = nn.LayerNorm(c.d_model, eps=1e-6, elementwise_affine=False)
        self.proj = nn.Linear(c.d_model, c.latent_channels)

    def forward(self, x, e):
        shift, scale = (self.table[None, None] + e[:, None, None]).unbind(2)
        return self.proj(self.norm(x) * (1 + scale) + shift)  # (batch, tokens, 128)


def unpack(tokens, grid):
    f, h, w = grid
    return tokens.reshape(tokens.shape[0], f, h, w, -1).permute(0, 4, 1, 2, 3)  # (batch, 128, F, H, W)


# %% [flow-matching] Flow matching
def sigmas(steps, tokens, base=(1024, 0.95), top=(4096, 2.05), terminal=0.1):
    """σ from 1 to 0 as for Wan, but the shift depends on the token count: μ grows linearly from 0.95 at
    1024 tokens to 2.05 at 4096, σ' = e^μ / (e^μ + 1/σ − 1). The schedule is then stretched to end at
    σ = 0.1 before the final jump to 0."""
    slope = (top[1] - base[1]) / (top[0] - base[0])
    mu = tokens * slope + (base[1] - slope * base[0])
    s = torch.linspace(1.0, 1 / steps, steps, dtype=torch.float64).float()  # float32 from here, as diffusers
    s = math.exp(mu) / (math.exp(mu) + (1 / s - 1))
    s = 1 - (1 - s) / ((1 - s[-1]) / (1 - terminal))
    return torch.cat([s, torch.zeros(1)])


def guided_velocity(v_text, v_negative, scale=3.0):
    return v_negative + scale * (v_text - v_negative)


def euler_step(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v


# %% [model] LTX-Video transformer
class LTXVideo(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.proj_in = nn.Linear(c.latent_channels, c.d_model)  # each latent position is a token
        self.conditioning = Conditioning(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)

    def forward(self, tokens, t, text, mask, grid, frame_rate=25):  # tokens: (batch, F · H · W, 128)
        cos, sin = rope(self.c, grid, (self.c.vae_temporal / frame_rate, self.c.vae_spatial, self.c.vae_spatial))
        e, modulation, text = self.conditioning(t, text)
        x = self.proj_in(tokens)
        for block in self.blocks:
            x = block(x, text, mask, modulation, cos, sin)
        return self.output(x, e)  # the velocity, one 128-vector per latent position
`;export{e as default};
