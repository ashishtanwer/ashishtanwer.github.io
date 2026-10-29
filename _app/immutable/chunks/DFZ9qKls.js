const e=`# FLUX.1-schnell (black-forest-labs/FLUX.1-schnell), the 12-billion-parameter MMDiT in the FLUX tab, written as plain
# PyTorch blocks. 19 double-stream blocks keep text and image weights apart, like SD 3's MMDiT; 38 single-stream
# blocks then run both token sets through one set of weights, attention and MLP side by side. Image tokens carry
# their row and column as a 2D rotary embedding. schnell is timestep-distilled: 1–4 flow-matching steps, no guidance.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 16  # the FLUX VAE's latent; 2 × 2 of it make one 64-number image token
    d_model: int = 3072
    n_heads: int = 24  # 3072 / 24 = 128 dimensions per head
    n_double: int = 19
    n_single: int = 38
    d_text: int = 4096  # T5 v1.1 XXL's width
    d_pooled: int = 768  # CLIP-L's pooled output
    rope_axes: tuple = (16, 56, 56)  # head dimensions turned by (index 0, row, column); 16 + 56 + 56 = 128
    rope_theta: float = 10000.0
    frequency_dim: int = 256  # cos and sin of t at 128 frequencies
    vae_scale: float = 0.3611  # latents are shifted and scaled to about unit variance
    vae_shift: float = 0.1159


# %% [text-encoders] CLIP and T5 text encoders
def encode_prompt(clip, t5, ids_clip, ids_t5):
    """CLIP-L gives one pooled vector: its last layer's state at the end-of-text token after the final LayerNorm,
    not projected. T5-XXL gives the 256 token states of its last layer. The pooled vector joins the timestep in c;
    the T5 tokens are the text stream. Unlike SD 3.5, no CLIP token states enter the transformer."""
    pooled = clip(ids_clip).pooler_output  # CLIPTextModel: (batch, 768)
    tokens = t5(ids_t5).last_hidden_state  # (batch, 256, 4096)
    return tokens, pooled


# %% [patch-embedding] Packing and image tokens
def pack(latent):
    """(batch, 16, H, W) → (batch, H/2 · W/2, 64): each 2 × 2 patch becomes one token, row by row; within a token
    the numbers run channel by channel, four pixels each."""
    b, c, h, w = latent.shape
    x = latent.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)  # (batch, row, col, channel, dy, dx)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(tokens, h, w):
    """The inverse of pack, back to an (h, w) latent."""
    b, _, d = tokens.shape
    x = tokens.view(b, h // 2, w // 2, d // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)  # (batch, channel, row, dy, col, dx)
    return x.reshape(b, d // 4, h, w)


class PatchEmbedding(nn.Module):
    """A Linear from each packed 64-number token to d_model. No position is added here: positions enter every
    attention as rotations (see rope)."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(4 * c.latent_channels, c.d_model)

    def forward(self, tokens):  # tokens: pack(latent), (batch, N, 64)
        return self.proj(tokens)


# %% [rope] 2D rotary position embedding
def position_ids(text_len, h, w):
    """Three numbers per token, text first: (0, 0, 0) for every text token, (0, row, column) for image tokens."""
    image = torch.zeros(h, w, 3)
    image[..., 1] = torch.arange(h)[:, None]
    image[..., 2] = torch.arange(w)[None, :]
    return torch.cat([torch.zeros(text_len, 3), image.view(h * w, 3)])


def rope_angles(ids, axes=(16, 56, 56), theta=10000.0):
    """Each head's 128 dimensions are 64 pairs. Pair i of axis a turns by ids[a] · θ^(−2i / dim_a): 8 pairs follow
    index 0 (always 0, so they never turn), 28 the row, 28 the column. Returns cos and sin, (tokens, 128) each."""
    cos, sin = [], []
    for axis, dim in enumerate(axes):
        frequencies = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        angles = ids[:, axis].double()[:, None] * frequencies  # (tokens, dim / 2)
        cos.append(angles.cos().repeat_interleave(2, dim=1))
        sin.append(angles.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos, dim=1).float(), torch.cat(sin, dim=1).float()


def rotate(x, cos, sin):
    """Turn each pair (x₂ᵢ, x₂ᵢ₊₁) by its angle. q · k then depends on how far apart two image tokens are in rows
    and in columns; text tokens sit at (0, 0, 0) and are not turned."""
    pairs = x.unflatten(-1, (-1, 2))  # (..., 64, 2)
    turned = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
    return x * cos + turned * sin


# %% [conditioning] Timestep and pooled-text conditioning
class Conditioning(nn.Module):
    """c = MLP(sin/cos of t) + MLP(pooled CLIP). schnell has no guidance input (FLUX.1-dev adds a third MLP)."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.time_in = nn.Linear(c.frequency_dim, c.d_model)
        self.time_out = nn.Linear(c.d_model, c.d_model)
        self.text_in = nn.Linear(c.d_pooled, c.d_model)
        self.text_out = nn.Linear(c.d_model, c.d_model)

    def sinusoid(self, t):  # t: (batch,) = σ × 1000
        half = self.frequency_dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        angles = t[:, None].float() * frequencies
        return torch.cat([angles.cos(), angles.sin()], dim=1)  # (batch, 256), cos first

    def forward(self, t, pooled):
        timestep = self.time_out(F.silu(self.time_in(self.sinusoid(t))))
        return timestep + self.text_out(F.silu(self.text_in(pooled)))  # c: (batch, d_model)


# %% [adaln] adaLN-Zero
class AdaLN(nn.Module):
    """From c, \`parts\` vectors of d_model: six per stream in a double block (shift β, scale γ, gate α for attention
    and for the MLP), three in a single block (one β, γ, α for both), two for the final layer (scale, then shift)."""

    def __init__(self, c: Config, parts=6):
        super().__init__()
        self.parts = parts
        self.linear = nn.Linear(c.d_model, parts * c.d_model)

    def forward(self, cond):  # cond: (batch, d_model)
        return self.linear(F.silu(cond))[:, None].chunk(self.parts, dim=-1)


def modulate(x, shift, scale):
    """LayerNorm without learned γ, β; the conditioning supplies them."""
    return F.layer_norm(x, x.shape[-1:], eps=1e-6) * (1 + scale) + shift


# %% [qk-norm] QK-norm
class RMSNorm(nn.Module):
    """Divide each head's query or key by its root mean square, then scale by a learned γ per head dimension."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):  # x: (..., head_dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# %% [joint-attention] Joint attention with RoPE
class Projections(nn.Module):
    """One set of query, key and value weights with QK-norm."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.norm_q = RMSNorm(c.d_model // c.n_heads)
        self.norm_k = RMSNorm(c.d_model // c.n_heads)

    def forward(self, x):  # x: (batch, tokens, d_model) → three (batch, tokens, heads, head_dim)
        q, k, v = (proj(x).unflatten(-1, (self.n_heads, -1)) for proj in (self.q, self.k, self.v))
        return self.norm_q(q), self.norm_k(k), v


def attend(q, k, v, cos, sin):
    """Rotate q and k by their tokens' positions, then plain softmax attention over every token."""
    q, k = rotate(q, cos[:, None], sin[:, None]), rotate(k, cos[:, None], sin[:, None])
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))  # (batch, heads, tokens, head_dim)
    weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
    return (weights @ v).transpose(1, 2).flatten(2)  # (batch, tokens, d_model)


class JointAttention(nn.Module):
    """Double blocks: text and image tokens projected by their own weights, then one attention over all of them
    (text first), split again into two output projections."""

    def __init__(self, c: Config):
        super().__init__()
        self.image = Projections(c)
        self.text = Projections(c)
        self.image_out = nn.Linear(c.d_model, c.d_model)
        self.text_out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x, y, cos, sin):  # x: image tokens (batch, N, d), y: text tokens (batch, M, d)
        m = y.shape[1]
        q, k, v = (torch.cat(pair, dim=1) for pair in zip(self.text(y), self.image(x)))
        heads = attend(q, k, v, cos, sin)
        return self.image_out(heads[:, m:]), self.text_out(heads[:, :m])


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, 4 * c.d_model)
        self.down = nn.Linear(4 * c.d_model, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [double-block] Double-stream block
class DoubleBlock(nn.Module):
    """Two streams with their own adaLN, projections and MLP, joined only inside the attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.image_ada = AdaLN(c)
        self.text_ada = AdaLN(c)
        self.attention = JointAttention(c)
        self.image_mlp = MLP(c)
        self.text_mlp = MLP(c)

    def forward(self, x, y, cond, cos, sin):  # x: image tokens, y: text tokens
        s1, c1, g1, s2, c2, g2 = self.image_ada(cond)
        ts1, tc1, tg1, ts2, tc2, tg2 = self.text_ada(cond)
        attended_x, attended_y = self.attention(modulate(x, s1, c1), modulate(y, ts1, tc1), cos, sin)
        x, y = x + g1 * attended_x, y + tg1 * attended_y
        x = x + g2 * self.image_mlp(modulate(x, s2, c2))
        return x, y + tg2 * self.text_mlp(modulate(y, ts2, tc2))


# %% [single-block] Single-stream block
class SingleBlock(nn.Module):
    """Text and image tokens as one sequence with one set of weights. One adaLN feeds an attention and an MLP side by
    side; their outputs are concatenated (d + 4d) and one Linear maps them back to d_model."""

    def __init__(self, c: Config):
        super().__init__()
        self.ada = AdaLN(c, 3)
        self.attention = Projections(c)
        self.mlp_in = nn.Linear(c.d_model, 4 * c.d_model)
        self.out = nn.Linear(5 * c.d_model, c.d_model)

    def forward(self, x, y, cond, cos, sin):
        m = y.shape[1]
        h = torch.cat([y, x], dim=1)  # (batch, M + N, d_model), text first
        shift, scale, gate = self.ada(cond)
        normed = modulate(h, shift, scale)
        attended = attend(*self.attention(normed), cos, sin)  # no output projection of its own
        mlp = F.gelu(self.mlp_in(normed), approximate="tanh")
        h = h + gate * self.out(torch.cat([attended, mlp], dim=-1))
        return h[:, m:], h[:, :m]  # image, text


# %% [output] Final layer
class FinalLayer(nn.Module):
    """adaLN (scale and shift, no gate) and a Linear back to 64 numbers per image token: the packed velocity."""

    def __init__(self, c: Config):
        super().__init__()
        self.ada = AdaLN(c, 2)
        self.linear = nn.Linear(c.d_model, 4 * c.latent_channels)

    def forward(self, x, cond):
        scale, shift = self.ada(cond)
        return self.linear(modulate(x, shift, scale))  # (batch, N, 64); unpack() makes it a latent


# %% [flow-matching] Flow matching and timestep distillation
# Training puts each latent on the straight line x_σ = (1 − σ) · x₀ + σ · ε and learns its velocity v = ε − x₀.
# schnell was then distilled (latent adversarial diffusion distillation) so that 1–4 Euler steps along that line
# land on a sharp image, with one model pass per step and no classifier-free guidance.
def noisy(x0, eps, sigma):
    """The forward path: a point on the line between the clean latent and the noise."""
    return (1 - sigma) * x0 + sigma * eps


def schnell_sigmas(steps=4):
    """σ evenly from 1 to 1/steps, then 0. schnell's scheduler has shift 1, so σ is not reshaped."""
    return torch.cat([torch.linspace(1.0, 1 / steps, steps, dtype=torch.float64).float(), torch.zeros(1)])


def euler_step(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v


def predicted_clean(x, v, sigma):
    """Where the line ends at σ = 0 if v held all the way: x̂₀ = x − σ · v."""
    return x - sigma * v


# %% [vae] VAE latents
# The VAE turns each 8 × 8 block of RGB pixels into one 16-channel latent vector (and back).
def to_latent(z, c: Config = Config()):
    return (z - c.vae_shift) * c.vae_scale


def from_latent(x, c: Config = Config()):
    return x / c.vae_scale + c.vae_shift


def encode(vae, image, c: Config = Config()):
    """RGB in [0, 1] → transformer latent, from the mean of the VAE's latent distribution."""
    return to_latent(vae.encode(image * 2 - 1).latent_dist.mode(), c)


def decode(vae, x, c: Config = Config()):
    """Transformer latent → RGB in [0, 1]."""
    return (vae.decode(from_latent(x, c)).sample / 2 + 0.5).clamp(0, 1)


# %% [model] Full model
class Flux(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.conditioning = Conditioning(c)
        self.context = nn.Linear(c.d_text, c.d_model)  # text tokens: 4096 → 3072
        self.double_blocks = nn.ModuleList(DoubleBlock(c) for _ in range(c.n_double))
        self.single_blocks = nn.ModuleList(SingleBlock(c) for _ in range(c.n_single))
        self.final = FinalLayer(c)

    def forward(self, tokens, t, text_tokens, pooled, h, w):  # tokens: pack(latent) on an h × w grid; t = σ × 1000
        cond = self.conditioning(t, pooled)
        x, y = self.patchify(tokens), self.context(text_tokens)
        cos, sin = rope_angles(position_ids(y.shape[1], h, w), self.c.rope_axes, self.c.rope_theta)
        for block in self.double_blocks:
            x, y = block(x, y, cond, cos, sin)
        for block in self.single_blocks:
            x, y = block(x, y, cond, cos, sin)
        return self.final(x, cond)  # packed velocity, (batch, h · w, 64)


@torch.no_grad()
def sample(model, tokens, pooled, generator, latent_size=128, steps=4):
    """1–4 Euler steps from pure noise, one model pass each."""
    c = model.c
    x = pack(torch.randn(1, c.latent_channels, latent_size, latent_size, generator=generator))
    sigmas = schnell_sigmas(steps)
    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        v = model(x, (sigma * 1000).expand(1), tokens, pooled, latent_size // 2, latent_size // 2)
        x = euler_step(x, v, sigma, sigma_next)
    return unpack(x, latent_size, latent_size)  # image = decode(vae, latent)


# latent = sample(Flux(), *encode_prompt(clip, t5, ids_clip, ids_t5), torch.Generator().manual_seed(0))
`;export{e as default};
