const e=`# Stable Diffusion 3.5 Medium (stabilityai/stable-diffusion-3.5-medium), the MMDiT in the SD 3.5 tab, written as
# plain PyTorch blocks. Text and image tokens keep separate weights but meet in one joint attention; a rectified-flow
# Euler sampler with classifier-free guidance walks a 16-channel VAE latent from noise to an image.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 16  # the SD3 VAE's latent; SD 1.5 and SDXL have 4
    patch_size: int = 2  # a 2 × 2 × 16 patch of latent is one image token
    d_model: int = 1536
    n_heads: int = 24  # 1536 / 24 = 64 dimensions per head
    n_layers: int = 24
    dual_attention_layers: tuple = tuple(range(13))  # MMDiT-X: blocks 0–12 add an image-only attention
    d_text: int = 4096  # T5-XXL's width; CLIP-L + CLIP-G (768 + 1280) are zero-padded to it
    d_pooled: int = 2048  # CLIP-L's pooled 768 next to CLIP-G's pooled 1280
    pos_max: int = 384  # the 2D sin-cos table covers 384 × 384 patches; an image uses its centre
    pos_step: float = 0.25  # table row i sits at position i · 0.25 − 40, so the centre is near 0
    pos_offset: float = -40.0
    frequency_dim: int = 256  # cos and sin of t at 128 frequencies
    shift: float = 3.0  # timestep shift of the flow-matching schedule
    vae_scale: float = 1.5305  # latents are shifted and scaled to about unit variance
    vae_shift: float = 0.0609


# %% [text-encoders] Three text encoders
def encode_prompt(clip_l, clip_g, t5, ids_l, ids_g, ids_t5, d_text=4096):
    """CLIP-L and CLIP-G each give the 77 token states of their second-to-last layer and one pooled, projected
    vector; T5-XXL gives the 256 token states of its last layer. The CLIP states sit side by side (768 + 1280),
    padded with zeros to T5's 4096, and come before T5's along the sequence: 77 + 256 = 333 text tokens."""
    out_l = clip_l(ids_l, output_hidden_states=True)  # CLIPTextModelWithProjection
    out_g = clip_g(ids_g, output_hidden_states=True)
    clip = torch.cat([out_l.hidden_states[-2], out_g.hidden_states[-2]], dim=-1)  # (batch, 77, 2048)
    clip = F.pad(clip, (0, d_text - clip.shape[-1]))  # (batch, 77, 4096)
    tokens = torch.cat([clip, t5(ids_t5).last_hidden_state], dim=1)  # (batch, 333, 4096)
    pooled = torch.cat([out_l.text_embeds, out_g.text_embeds], dim=-1)  # (batch, 2048)
    return tokens, pooled


# %% [patch-embedding] Patch embedding
def sincos(d, positions):
    """d/2 sines then d/2 cosines of each position at frequencies 10000^(−2i/d)."""
    frequencies = 10000 ** (-torch.arange(d // 2, dtype=torch.float64) / (d / 2))
    angles = positions[:, None].double() * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class PatchEmbedding(nn.Module):
    """Cut the latent into 2 × 2 patches, project each to d_model, add a fixed 2D sin-cos position. The positions
    come from the centre of a 384 × 384 table, so every image size up to 768 latent pixels finds its own. The
    checkpoint ships the table (pos_embed.pos_embed, bf16); it is this sin-cos of i/4 − 40, rounded to bf16."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.proj = nn.Conv2d(c.latent_channels, c.d_model, kernel_size=c.patch_size, stride=c.patch_size)

    def position(self, h, w):
        c = self.c
        top, left = (c.pos_max - h) // 2, (c.pos_max - w) // 2  # the centre of the table
        rows = torch.arange(top, top + h, dtype=torch.float32) * c.pos_step + c.pos_offset
        cols = torch.arange(left, left + w, dtype=torch.float32) * c.pos_step + c.pos_offset
        rows, cols = torch.meshgrid(rows, cols, indexing="ij")
        # First half of each vector encodes the patch's column, second half its row; not learned.
        return torch.cat([sincos(c.d_model // 2, cols.flatten()), sincos(c.d_model // 2, rows.flatten())], 1).float()

    def forward(self, latent):  # latent: (batch, 16, H, W)
        patches = self.proj(latent).flatten(2).transpose(1, 2)  # (batch, H/2 · W/2, d_model), row by row
        p = self.c.patch_size
        return patches + self.position(latent.shape[-2] // p, latent.shape[-1] // p)


# %% [conditioning] Timestep and pooled-text conditioning
class Conditioning(nn.Module):
    """c = MLP(sin/cos of t) + MLP(pooled CLIP text). One vector per image; every block's adaLN reads it."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.time_in = nn.Linear(c.frequency_dim, c.d_model)
        self.time_out = nn.Linear(c.d_model, c.d_model)
        self.text_in = nn.Linear(c.d_pooled, c.d_model)
        self.text_out = nn.Linear(c.d_model, c.d_model)

    def sinusoid(self, t):  # t: (batch,) = σ × 1000, from 1000 (noise) down toward 0
        half = self.frequency_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        angles = t[:, None].float() * frequencies
        return torch.cat([angles.cos(), angles.sin()], dim=1)  # (batch, 256), cos first

    def forward(self, t, pooled):
        timestep = self.time_out(F.silu(self.time_in(self.sinusoid(t))))
        text = self.text_out(F.silu(self.text_in(pooled)))
        return timestep + text  # c: (batch, d_model)


# %% [adaln] adaLN-Zero
class AdaLN(nn.Module):
    """From c, \`parts\` vectors of d_model. A stream's six are shift β, scale γ and gate α for its attention and its
    MLP; MMDiT-X's image stream adds β, γ, α for its second attention (nine). The final layer and the last block's
    text stream take two, scale then shift, and no gate."""

    def __init__(self, c: Config, parts=6):
        super().__init__()
        self.parts = parts
        self.linear = nn.Linear(c.d_model, parts * c.d_model)

    def forward(self, cond):  # cond: (batch, d_model)
        return self.linear(F.silu(cond))[:, None].chunk(self.parts, dim=-1)  # \`parts\` × (batch, 1, d_model)


def modulate(x, shift, scale):
    """LayerNorm without learned γ, β; the conditioning supplies them."""
    return F.layer_norm(x, x.shape[-1:], eps=1e-6) * (1 + scale) + shift


# %% [qk-norm] QK-norm
class RMSNorm(nn.Module):
    """Divide each head's query or key by its root mean square, then scale by a learned γ per head dimension.
    It bounds q · k, which kept growing in large MMDiTs and made half-precision training diverge."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):  # x: (..., head_dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# %% [joint-attention] Joint attention
class Projections(nn.Module):
    """One stream's own query, key and value weights, and its own QK-norm."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.norm_q = RMSNorm(c.d_model // c.n_heads)
        self.norm_k = RMSNorm(c.d_model // c.n_heads)

    def forward(self, x):  # x: (batch, tokens, d_model) → three (batch, heads, tokens, head_dim)
        b, t, _ = x.shape
        q, k, v = (proj(x).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        return self.norm_q(q), self.norm_k(k), v


def attend(q, k, v):
    weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # no mask: everyone sees everyone
    return (weights @ v).transpose(1, 2).flatten(2)  # (batch, tokens, d_model)


class JointAttention(nn.Module):
    """Image and text tokens are projected by separate weights but share one softmax: each of the 256 image and
    333 text tokens attends to all 589. The result is split again, and each stream has its own output projection."""

    def __init__(self, c: Config, text_out=True):
        super().__init__()
        self.image = Projections(c)
        self.text = Projections(c)
        self.image_out = nn.Linear(c.d_model, c.d_model)
        self.text_out = nn.Linear(c.d_model, c.d_model) if text_out else None  # the last block drops the text

    def forward(self, x, y):  # x: image tokens (batch, N, d_model), y: text tokens (batch, M, d_model)
        n = x.shape[1]
        q, k, v = (torch.cat(pair, dim=2) for pair in zip(self.image(x), self.text(y)))  # image first, then text
        heads = attend(q, k, v)  # (batch, N + M, d_model)
        image, text = heads[:, :n], heads[:, n:]
        return self.image_out(image), self.text_out(text) if self.text_out else None


class ImageAttention(nn.Module):
    """MMDiT-X (blocks 0–12 of SD 3.5 Medium): a second attention over the image tokens alone, with its own weights."""

    def __init__(self, c: Config):
        super().__init__()
        self.image = Projections(c)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):
        return self.out(attend(*self.image(x)))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, 4 * c.d_model)
        self.down = nn.Linear(4 * c.d_model, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] MMDiT block
class Block(nn.Module):
    """Two streams with their own adaLN, projections and MLP, joined only inside the attention.
    x: the image tokens, y: the text tokens, c: the conditioning vector."""

    def __init__(self, c: Config, index: int):
        super().__init__()
        self.dual = index in c.dual_attention_layers
        self.last = index == c.n_layers - 1  # nothing reads the text stream after the last block
        self.image_ada = AdaLN(c, 9 if self.dual else 6)
        self.text_ada = AdaLN(c, 2 if self.last else 6)
        self.attention = JointAttention(c, text_out=not self.last)
        self.image_attention = ImageAttention(c) if self.dual else None
        self.image_mlp = MLP(c)
        self.text_mlp = None if self.last else MLP(c)

    def forward(self, x, y, cond):
        image = self.image_ada(cond)
        shift1, scale1, gate1, shift2, scale2, gate2 = image[:6]
        if self.last:
            scale, shift = self.text_ada(cond)
            y_in = modulate(y, shift, scale)
        else:
            t_shift1, t_scale1, t_gate1, t_shift2, t_scale2, t_gate2 = self.text_ada(cond)
            y_in = modulate(y, t_shift1, t_scale1)
        attended_x, attended_y = self.attention(modulate(x, shift1, scale1), y_in)
        out = x + gate1 * attended_x
        if self.dual:  # its input is the block's input x, modulated by the last three vectors
            shift3, scale3, gate3 = image[6:]
            out = out + gate3 * self.image_attention(modulate(x, shift3, scale3))
        out = out + gate2 * self.image_mlp(modulate(out, shift2, scale2))
        if self.last:
            return out, None
        y = y + t_gate1 * attended_y
        return out, y + t_gate2 * self.text_mlp(modulate(y, t_shift2, t_scale2))


# %% [output] Final layer and unpatchify
class FinalLayer(nn.Module):
    """adaLN (scale and shift, no gate), a Linear to 2 × 2 × 16 numbers per image token, then patches back to a grid.
    The result is the velocity v, the same shape as the latent."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.ada = AdaLN(c, 2)
        self.linear = nn.Linear(c.d_model, c.patch_size * c.patch_size * c.latent_channels)

    def forward(self, x, cond, h, w):  # x: (batch, h · w, d_model) for an h × w grid of tokens
        scale, shift = self.ada(cond)
        x = self.linear(modulate(x, shift, scale))  # (batch, h · w, 64)
        p, channels = self.c.patch_size, self.c.latent_channels
        x = x.view(-1, h, w, p, p, channels)  # (batch, row, col, dy, dx, channel)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (batch, channel, row, dy, col, dx)
        return x.reshape(-1, channels, h * p, w * p)


# %% [flow-matching] Rectified flow and the Euler sampler
# Training puts each latent on the straight line x_σ = (1 − σ) · x₀ + σ · ε from the image (σ = 0) to noise
# (σ = 1), and the model learns the line's velocity v = ε − x₀. Sampling walks the line backwards with Euler steps.
def shift_sigma(sigma, shift=3.0):
    """σ' = s·σ / (1 + (s − 1)·σ): with s = 3, more of the steps are spent at high noise."""
    return shift * sigma / (1 + (shift - 1) * sigma)


def flow_sigmas(steps, shift=3.0, train_steps=1000):
    """The σ of each of \`steps\` steps, then 0. Like the reference scheduler, it spaces them evenly between the shifted
    ends of the 1000 training levels (1 and shift(1/1000)), then shifts those values once more."""
    lowest = shift_sigma(torch.tensor(1 / train_steps), shift).item()
    sigmas = shift_sigma(torch.linspace(1.0, lowest, steps, dtype=torch.float64), shift).float()
    return torch.cat([sigmas, torch.zeros(1)])  # (steps + 1,); the timestep the model sees is σ × 1000


def euler_step(x, v, sigma, sigma_next):
    """One straight step along the predicted velocity; σ_next < σ, so it moves toward the image."""
    return x + (sigma_next - sigma) * v


def predicted_clean(x, v, sigma):
    """Where the line ends at σ = 0 if v held all the way: x̂₀ = x − σ · v."""
    return x - sigma * v


# %% [guidance] Classifier-free guidance
def guide(v_text, v_null, scale=4.5):
    """Run the prompt and the empty prompt side by side; push the velocity away from the empty prompt's."""
    return v_null + scale * (v_text - v_null)


# %% [vae] VAE latents
# The VAE turns each 8 × 8 block of RGB pixels into one 16-channel latent vector (and back). Its latents are
# shifted and scaled to about unit variance before the transformer sees them.
def to_latent(z, c: Config = Config()):
    return (z - c.vae_shift) * c.vae_scale


def from_latent(x, c: Config = Config()):
    return x / c.vae_scale + c.vae_shift


def decode(vae, x, c: Config = Config()):
    """Final latent → RGB in [0, 1]."""
    return (vae.decode(from_latent(x, c)).sample / 2 + 0.5).clamp(0, 1)


# %% [model] Full model
class SD35(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.conditioning = Conditioning(c)
        self.context = nn.Linear(c.d_text, c.d_model)  # text tokens: 4096 → 1536
        self.blocks = nn.ModuleList(Block(c, i) for i in range(c.n_layers))
        self.final = FinalLayer(c)

    def forward(self, latent, t, text_tokens, pooled):  # latent: (batch, 16, H, W); t = σ × 1000
        cond = self.conditioning(t, pooled)
        x, y = self.patchify(latent), self.context(text_tokens)
        for block in self.blocks:
            x, y = block(x, y, cond)
        p = self.c.patch_size
        return self.final(x, cond, latent.shape[-2] // p, latent.shape[-1] // p)  # v: (batch, 16, H, W)


@torch.no_grad()
def sample(model, tokens, pooled, null_tokens, null_pooled, generator, latent_size=128, steps=40, guidance=4.5):
    """Euler steps from pure noise, each running the prompt and the empty prompt side by side."""
    c = model.c
    x = torch.randn(1, c.latent_channels, latent_size, latent_size, generator=generator)
    sigmas = flow_sigmas(steps, c.shift)
    text, pool = torch.cat([null_tokens, tokens]), torch.cat([null_pooled, pooled])  # empty prompt, then prompt
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        v_null, v_text = model(torch.cat([x, x]), (sigma * 1000).expand(2), text, pool).chunk(2)
        x = euler_step(x, guide(v_text, v_null, guidance), sigma, sigma_next)
    return x  # image = decode(vae, x)


# latent = sample(SD35(), *encode_prompt(...), *encode_prompt(... "" ...), torch.Generator().manual_seed(0))
`;export{e as default};
