const n=`# Sana 600M 512 px, the linear-attention diffusion transformer in the Sana tab, written as plain PyTorch blocks.
# Gemma-2 2B reads the prompt; 28 blocks with ReLU linear attention and a convolutional Mix-FFN predict the flow
# velocity of a 32 × 16 × 16 latent; DPM-Solver++ takes 20 steps on flow sigmas and the 32× DC-AE decodes 512 px.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # Gemma-2 2B, used as a text encoder
    vocab: int = 256000
    text_dim: int = 2304
    text_heads: int = 8
    text_kv_heads: int = 4  # two query heads share each key/value head
    text_head_dim: int = 256
    text_ff: int = 9216
    text_layers: int = 26
    query_scalar: float = 256.0  # scores are scaled by 1/√256
    softcap: float = 50.0  # scores pass through 50 · tanh(s / 50)
    sliding_window: int = 4096  # on every other layer; longer than any prompt here
    rope_theta: float = 10000.0
    max_tokens: int = 300  # slot 0 and the last 299 slots are kept
    # The diffusion transformer
    latent_channels: int = 32
    latent_size: int = 16  # 512 px / the DC-AE's factor of 32
    d_model: int = 1152
    n_heads: int = 36  # linear attention: 36 heads of 32
    cross_heads: int = 16  # cross-attention: 16 heads of 72
    n_layers: int = 28
    mlp_ratio: float = 2.5  # Mix-FFN: 2 × 2880 channels before the gate
    frequency_dim: int = 256
    flow_shift: float = 3.0
    # The DC-AE decoder (f32c32)
    vae_channels: tuple = (128, 256, 512, 512, 1024, 1024)
    vae_attention: tuple = (False, False, False, True, True, True)  # EfficientViT blocks in the three deepest stages
    vae_layers: int = 3
    vae_head_dim: int = 32
    vae_scaling: float = 0.41407


# %% [text-encoder] Gemma-2 text encoder
class GemmaNorm(nn.Module):
    """RMSNorm with a (1 + w) scale: the weights start at 0."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1 + self.weight)


def rope(x, theta=10000.0):
    """Rotate each head's halves by position · θ^(−2i/d); x: (batch, heads, n, d)."""
    n, d = x.shape[-2], x.shape[-1]
    angles = torch.arange(n, dtype=torch.float32)[:, None] * theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
    half = d // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], -1) * sin


class GemmaLayer(nn.Module):
    def __init__(self, c: Config, sliding: bool):
        super().__init__()
        self.c, self.sliding = c, sliding
        self.input_norm, self.post_attention_norm = GemmaNorm(c.text_dim), GemmaNorm(c.text_dim)
        self.pre_ff_norm, self.post_ff_norm = GemmaNorm(c.text_dim), GemmaNorm(c.text_dim)
        self.q = nn.Linear(c.text_dim, c.text_heads * c.text_head_dim, bias=False)
        self.k = nn.Linear(c.text_dim, c.text_kv_heads * c.text_head_dim, bias=False)
        self.v = nn.Linear(c.text_dim, c.text_kv_heads * c.text_head_dim, bias=False)
        self.o = nn.Linear(c.text_heads * c.text_head_dim, c.text_dim, bias=False)
        self.gate = nn.Linear(c.text_dim, c.text_ff, bias=False)
        self.up = nn.Linear(c.text_dim, c.text_ff, bias=False)
        self.down = nn.Linear(c.text_ff, c.text_dim, bias=False)

    def forward(self, x, mask):  # mask: (batch, n), 1 on tokens, 0 on padding
        c, n = self.c, x.shape[1]
        h = self.input_norm(x)
        q = rope(self.q(h).unflatten(-1, (c.text_heads, -1)).transpose(1, 2), c.rope_theta)
        k = rope(self.k(h).unflatten(-1, (c.text_kv_heads, -1)).transpose(1, 2), c.rope_theta)
        v = self.v(h).unflatten(-1, (c.text_kv_heads, -1)).transpose(1, 2)
        k, v = (t.repeat_interleave(c.text_heads // c.text_kv_heads, dim=1) for t in (k, v))
        scores = q @ k.transpose(-1, -2) / math.sqrt(c.query_scalar)
        scores = c.softcap * torch.tanh(scores / c.softcap)
        i, j = torch.arange(n)[:, None], torch.arange(n)[None]
        allowed = (j <= i) & mask[:, None, None, :].bool()  # causal: Gemma is a decoder, used as an encoder
        if self.sliding:
            allowed = allowed & (i - j < c.sliding_window)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        x = x + self.post_attention_norm(self.o((scores.softmax(-1) @ v).transpose(1, 2).flatten(2)))
        h = self.pre_ff_norm(x)
        return x + self.post_ff_norm(self.down(F.gelu(self.gate(h), approximate="tanh") * self.up(h)))


class GemmaEncoder(nn.Module):
    """Token ids → one 2304-wide vector per token. Sana's pipeline prepends a long "complex human instruction"
    (a description of how to enrich prompts), runs all of it and keeps slot 0 and the last 299 slots."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.text_dim)
        self.layers = nn.ModuleList(GemmaLayer(c, sliding=i % 2 == 0) for i in range(c.text_layers))
        self.norm = GemmaNorm(c.text_dim)

    def forward(self, ids, mask):
        x = self.embed(ids) * math.sqrt(self.c.text_dim)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def keep_slots(embeds, mask, kept=300):
    """Slot 0 (<bos>) and the last kept − 1 slots: the instruction's tail, the prompt and its padding."""
    index = [0] + list(range(embeds.shape[1] - kept + 1, embeds.shape[1]))
    return embeds[:, index], mask[:, index]


# %% [caption-projection] Caption projection and norm
class CaptionProjection(nn.Module):
    """Gemma's 2304-wide token vectors → y, 1152 wide, then RMSNorm: the cross-attention's keys and values."""

    def __init__(self, c: Config):
        super().__init__()
        self.linear_1 = nn.Linear(c.text_dim, c.d_model)
        self.linear_2 = nn.Linear(c.d_model, c.d_model)
        self.norm_weight = nn.Parameter(torch.ones(c.d_model))

    def forward(self, text):
        y = self.linear_2(F.gelu(self.linear_1(text), approximate="tanh"))
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5) * self.norm_weight


# %% [patchify] Patch embedding
class PatchEmbedding(nn.Module):
    """One token per latent pixel: a 1 × 1 conv lifts its 32 channels to 1152. No position embedding at all."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Conv2d(c.latent_channels, c.d_model, kernel_size=1)

    def forward(self, latent):  # (batch, 32, 16, 16)
        return self.proj(latent).flatten(2).transpose(1, 2)  # (batch, 256, 1152), row by row


# %% [adaln-single] adaLN-single
def sinusoid(t, d=256):
    half = d // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    angles = t[:, None].float() * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class AdaLNSingle(nn.Module):
    """As in PixArt-α: one timestep MLP and one Linear shared by all 28 blocks; each block adds a 6 × 1152 table."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.mlp_in = nn.Linear(c.frequency_dim, c.d_model)
        self.mlp_out = nn.Linear(c.d_model, c.d_model)
        self.linear = nn.Linear(c.d_model, 6 * c.d_model)

    def forward(self, t):  # t = ⌊1000 σ⌋
        t0 = self.mlp_out(F.silu(self.mlp_in(sinusoid(t, self.frequency_dim))))
        return self.linear(F.silu(t0)), t0


def modulation(shared, table):
    return (table[None] + shared.unflatten(1, (6, -1))).chunk(6, dim=1)  # β₁ γ₁ α₁ β₂ γ₂ α₂, each (batch, 1, 1152)


def layer_norm(x):
    return F.layer_norm(x, x.shape[-1:], eps=1e-6)


# %% [linear-attention] ReLU linear attention
class LinearAttention(nn.Module):
    """No softmax: a token's weight on another is relu(q)·relu(k), divided by its row's sum. Summed over the keys
    first, each head needs one 32 × 32 state S = Σ v relu(k)ᵀ and z = Σ relu(k), so the cost grows with n, not n²."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.n_heads
        self.q, self.k, self.v = (nn.Linear(c.d_model, c.d_model, bias=False) for _ in range(3))
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):
        q, k, v = (p(x).unflatten(-1, (self.heads, -1)).transpose(1, 2) for p in (self.q, self.k, self.v))
        q, k = F.relu(q), F.relu(k)  # (batch, 36, n, 32)
        state = v.transpose(-1, -2) @ k  # S: (batch, 36, 32, 32)
        z = k.sum(dim=2)  # (batch, 36, 32)
        o = (q @ state.transpose(-1, -2)) / (q @ z[..., None] + 1e-15)
        return self.out(o.transpose(1, 2).flatten(2))


def mixing_matrix(q, k):
    """The same weights written out as an n × n matrix for one head (what the head card draws): rows sum to 1."""
    scores = F.relu(q) @ F.relu(k).transpose(-1, -2)
    return scores / (scores.sum(-1, keepdim=True) + 1e-15)


# %% [cross-attention] Cross-attention to the prompt
class CrossAttention(nn.Module):
    """Softmax attention, 16 heads of 72: every token's query reads the prompt tokens y; padding gets −10000."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.cross_heads
        self.q, self.k, self.v, self.out = (nn.Linear(c.d_model, c.d_model) for _ in range(4))

    def forward(self, x, y, mask):
        q = self.q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = self.k(y).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        v = self.v(y).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + (1 - mask.float())[:, None, None] * -10000.0
        return self.out((scores.softmax(-1) @ v).transpose(1, 2).flatten(2))


# %% [mix-ffn] Mix-FFN
class MixFFN(nn.Module):
    """The MLP works on the 16 × 16 grid: a 1 × 1 conv to 2 × 2880 channels and SiLU, a 3 × 3 depthwise conv that
    mixes each channel with its 8 neighbours (Sana's only position signal), half the channels gating the other half,
    and a 1 × 1 conv back to 1152. The DC-AE's EfficientViT blocks reuse it with an RMSNorm and a residual."""

    def __init__(self, channels, expand, norm=False, residual=False):
        super().__init__()
        hidden = int(expand * channels)
        self.inverted = nn.Conv2d(channels, 2 * hidden, 1)
        self.depthwise = nn.Conv2d(2 * hidden, 2 * hidden, 3, padding=1, groups=2 * hidden)
        self.point = nn.Conv2d(hidden, channels, 1, bias=False)
        self.norm = ChannelNorm(channels) if norm else None
        self.residual = residual

    def forward(self, x):  # (batch, channels, height, width)
        h = self.depthwise(F.silu(self.inverted(x)))
        value, gate = h.chunk(2, dim=1)
        h = self.point(value * F.silu(gate))
        if self.norm is not None:
            h = self.norm(h)
        return x + h if self.residual else h


class ChannelNorm(nn.Module):
    """RMSNorm over the channels of each pixel, with a weight and a bias."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        x = x.movedim(1, -1)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight + self.bias
        return x.movedim(-1, 1)


# %% [block] Sana block
class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attention = LinearAttention(c)
        self.cross = CrossAttention(c)
        self.mlp = MixFFN(c.d_model, c.mlp_ratio)
        self.table = nn.Parameter(torch.zeros(6, c.d_model))

    def forward(self, x, shared, y, mask, side):
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation(shared, self.table)
        x = x + gate1 * self.attention(layer_norm(x) * (1 + scale1) + shift1)
        x = x + self.cross(x, y, mask)  # no norm and no gate, as in PixArt-α
        h = layer_norm(x) * (1 + scale2) + shift2
        h = self.mlp(h.unflatten(1, (side, side)).permute(0, 3, 1, 2))  # tokens back onto the grid for the convs
        return x + gate2 * h.flatten(2).transpose(1, 2)


# %% [final-layer] Final layer
class FinalLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(2, c.d_model))
        self.linear = nn.Linear(c.d_model, c.latent_channels)

    def forward(self, x, t0, side):
        shift, scale = (self.table[None] + t0[:, None]).chunk(2, dim=1)
        x = self.linear(layer_norm(x) * (1 + scale) + shift)  # (batch, 256, 32)
        return x.transpose(1, 2).unflatten(2, (side, side))  # (batch, 32, 16, 16): the velocity v̂


# %% [flow-step] Guidance and the flow-matching step
def flow_schedule(steps, shift=3.0, train_steps=1000):
    """σ from 1 towards 0, shifted towards the noisy end: σ' = 3σ / (1 + 2σ). On the straight path
    x_σ = (1 − σ) x₀ + σ ε the model predicts the velocity v = ε − x₀; the transformer sees t = ⌊1000 σ⌋."""
    sigma = 1 - torch.linspace(1, 1 / train_steps, steps + 1, dtype=torch.float64)
    sigma = (shift * sigma / (1 + (shift - 1) * sigma)).flip(0)[:-1]
    return (sigma * train_steps).long(), torch.cat([sigma, torch.zeros(1, dtype=torch.float64)]).float()


def guided_velocity(v_unconditional, v_conditional, scale=4.5):
    return v_unconditional + scale * (v_conditional - v_unconditional)


def flow_step(x, v, i, sigma, previous_x0=None):
    """x̂₀ = x − σ v̂, then DPM-Solver++ with α = 1 − σ: second order (midpoint) between the first and last steps."""
    x0 = x - sigma[i] * v
    if sigma[i + 1] == 0:
        return x0, x0
    lam = lambda j: torch.log((1 - sigma[j]) / sigma[j])  # log(α/σ) with α = 1 − σ  # noqa: E731
    h = lam(i + 1) - lam(i)
    alpha_next = 1 - sigma[i + 1]
    x_next = (sigma[i + 1] / sigma[i]) * x - alpha_next * (torch.exp(-h) - 1) * x0
    if previous_x0 is not None:
        r = (lam(i) - lam(i - 1)) / h
        x_next = x_next - 0.5 * alpha_next * (torch.exp(-h) - 1) * (x0 - previous_x0) / r
    return x_next, x0


# %% [dc-ae-decode] DC-AE decoder (32×)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = ChannelNorm(channels)

    def forward(self, x):
        return x + self.norm(self.conv2(F.silu(self.conv1(x))))


class MultiscaleLinearAttention(nn.Module):
    """EfficientViT's attention: q, k, v per pixel, and a second copy blurred by a 5 × 5 depthwise conv; ReLU linear
    attention over all pixels for both, heads of 32 channels."""

    def __init__(self, channels, head_dim):
        super().__init__()
        self.head_dim = head_dim
        heads = channels // head_dim
        self.q, self.k, self.v = (nn.Linear(channels, channels, bias=False) for _ in range(3))
        self.blur = nn.Conv2d(3 * channels, 3 * channels, 5, padding=2, groups=3 * channels, bias=False)
        self.mix = nn.Conv2d(3 * channels, 3 * channels, 1, groups=3 * heads, bias=False)
        self.out = nn.Linear(2 * channels, channels, bias=False)
        self.norm = ChannelNorm(channels)

    def forward(self, x):
        b, _, height, width = x.shape
        pixels = x.movedim(1, -1)
        qkv = torch.cat([self.q(pixels), self.k(pixels), self.v(pixels)], dim=-1).movedim(-1, 1)
        qkv = torch.cat([qkv, self.mix(self.blur(qkv))], dim=1)  # (batch, 6 · channels, h, w)
        qkv = qkv.reshape(b, -1, 3 * self.head_dim, height * width)
        q, k, v = qkv.chunk(3, dim=2)  # (batch, heads, 32, pixels)
        q, k = F.relu(q), F.relu(k)
        v = F.pad(v, (0, 0, 0, 1), value=1.0)  # a row of ones carries the normaliser Σ k·q
        o = (v @ k.transpose(-1, -2)) @ q
        o = (o[:, :, :-1] / (o[:, :, -1:] + 1e-15)).reshape(b, -1, height, width)
        return x + self.norm(self.out(o.movedim(1, -1)).movedim(-1, 1))


class EfficientViTBlock(nn.Module):
    def __init__(self, channels, head_dim):
        super().__init__()
        self.attention = MultiscaleLinearAttention(channels, head_dim)
        self.conv = MixFFN(channels, 4, norm=True, residual=True)

    def forward(self, x):
        return self.conv(self.attention(x))


class Upsample(nn.Module):
    """Nearest 2× then a 3 × 3 conv, plus a shortcut that spreads the input's channels over the 2 × 2 pixels."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.repeats = c_out * 4 // c_in

    def forward(self, x):
        shortcut = F.pixel_shuffle(x.repeat_interleave(self.repeats, dim=1), 2)
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest")) + shortcut


class DCAEDecoder(nn.Module):
    """32 × 16 × 16 → 3 × 512 × 512: five 2× upsamplings, from 1024 channels down to 128."""

    def __init__(self, c: Config):
        super().__init__()
        widths = c.vae_channels
        self.scaling = c.vae_scaling
        self.conv_in = nn.Conv2d(c.latent_channels, widths[-1], 3, padding=1)
        self.repeats = widths[-1] // c.latent_channels
        self.stages = nn.ModuleList()
        for i in reversed(range(len(widths))):
            stage = []
            if i < len(widths) - 1:
                stage.append(Upsample(widths[i + 1], widths[i]))
            make = (lambda w: EfficientViTBlock(w, c.vae_head_dim)) if c.vae_attention[i] else ResBlock
            stage += [make(widths[i]) for _ in range(c.vae_layers)]
            self.stages.append(nn.Sequential(*stage))
        self.norm = ChannelNorm(widths[0])
        self.conv_out = nn.Conv2d(widths[0], 3, 3, padding=1)

    def forward(self, latent):
        z = latent / self.scaling
        x = self.conv_in(z) + z.repeat_interleave(self.repeats, dim=1)
        for stage in self.stages:
            x = stage(x)
        image = self.conv_out(F.relu(self.norm(x)))
        return (image / 2 + 0.5).clamp(0, 1)


# %% [model] Full model
class Sana(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.adaln = AdaLNSingle(c)
        self.caption = CaptionProjection(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.final = FinalLayer(c)

    def forward(self, latent, t, text, mask):  # text: Gemma output (batch, 300, 2304)
        side = latent.shape[-1]
        x = self.patchify(latent)
        shared, t0 = self.adaln(t)
        y = self.caption(text)
        for block in self.blocks:
            x = block(x, shared, y, mask, side)
        return self.final(x, t0, side)


@torch.no_grad()
def generate(model, vae, text, mask, null_text, null_mask, steps=20, scale=4.5, generator=None):
    c = model.c
    latent = torch.randn(1, c.latent_channels, c.latent_size, c.latent_size, generator=generator)
    t, sigma = flow_schedule(steps, c.flow_shift)
    texts, masks = torch.cat([null_text, text]), torch.cat([null_mask, mask])
    previous = None
    for i in range(steps):
        v = guided_velocity(*model(torch.cat([latent, latent]), t[i].expand(2), texts, masks).chunk(2), scale)
        latent, x0 = flow_step(latent, v, i, sigma, previous if 0 < i < steps - 1 else None)
        previous = x0
    return latent, vae(latent)
`;export{n as default};
