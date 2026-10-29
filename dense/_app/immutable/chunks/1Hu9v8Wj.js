const e=`# Cosmos-Predict2 2B Video2World (nvidia/Cosmos-Predict2-2B-Video2World), the world foundation model in the Cosmos
# tab, written as plain PyTorch blocks. Given the first frame(s) of a video and a prompt, it predicts the frames
# that follow: a 2B DiT denoises the Wan 2.1 VAE latents of the whole clip, with the given frames held clean, and
# reads the prompt through cross-attention to T5-11B's encoder. Sampling: Karras σ (EDM) with rectified-flow
# preconditioning and classifier-free guidance. Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 16  # Wan 2.1 VAE: 8 × 8 pixels and 4 video frames per latent pixel
    patch: tuple = (1, 2, 2)
    d_model: int = 2048
    n_heads: int = 16  # 128 dimensions per head
    n_layers: int = 28
    d_mlp: int = 8192
    adaln_lora: int = 256
    text_dim: int = 1024  # T5-11B encoder states
    rope_scale: tuple = (1.0, 3.0, 3.0)  # NTK scaling of the frame, row and column frequencies
    base_fps: int = 24
    # T5-11B's encoder (Cosmos keeps 512 tokens, the padding set to zero)
    vocab: int = 32128
    t5_dim: int = 1024
    t5_heads: int = 128
    t5_head_dim: int = 128
    t5_mlp: int = 65536
    t5_layers: int = 24
    t5_buckets: int = 32
    t5_max_distance: int = 128


# %% [text-encoder] T5 text encoder
class T5Block(nn.Module):
    """Pre-norm (RMS, no mean) self-attention with a learned relative-position bias and no 1/√d, then a ReLU MLP."""

    def __init__(self, c: Config):
        super().__init__()
        inner = c.t5_heads * c.t5_head_dim
        self.heads = c.t5_heads
        self.norm1 = nn.Parameter(torch.ones(c.t5_dim))
        self.q, self.k, self.v = (nn.Linear(c.t5_dim, inner, bias=False) for _ in range(3))
        self.o = nn.Linear(inner, c.t5_dim, bias=False)
        self.norm2 = nn.Parameter(torch.ones(c.t5_dim))
        self.wi = nn.Linear(c.t5_dim, c.t5_mlp, bias=False)
        self.wo = nn.Linear(c.t5_mlp, c.t5_dim, bias=False)

    @staticmethod
    def rms(x, weight):
        return weight * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).type_as(x)

    def forward(self, x, bias):  # bias: (1, heads, n, n), position bias plus the padding mask
        b, n, _ = x.shape
        h = self.rms(x, self.norm1)
        q, k, v = (proj(h).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = (q @ k.transpose(-1, -2) + bias).float().softmax(-1).type_as(q)
        x = x + self.o((weights @ v).transpose(1, 2).reshape(b, n, -1))
        return x + self.wo(F.relu(self.wi(self.rms(x, self.norm2))))


class T5Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab, c.t5_dim)
        self.relative = nn.Embedding(c.t5_buckets, c.t5_heads)  # block 1's, shared by every block
        self.blocks = nn.ModuleList(T5Block(c) for _ in range(c.t5_layers))
        self.norm = nn.Parameter(torch.ones(c.t5_dim))

    def bucket(self, distance):
        """Half the buckets for keys before the query, half after; exact up to 8 apart, then log-spaced to 128."""
        half = self.c.t5_buckets // 2
        exact = half // 2
        bucket = (distance > 0).long() * half
        d = distance.abs()
        far = exact + (torch.log(d.float() / exact) / math.log(self.c.t5_max_distance / exact) * (half - exact)).long()
        return bucket + torch.where(d < exact, d, far.clamp(max=half - 1))

    def forward(self, ids, mask):  # ids, mask: (batch, 512)
        n = ids.shape[1]
        positions = torch.arange(n)
        bias = self.relative(self.bucket(positions[None, :] - positions[:, None])).permute(2, 0, 1)[None]
        bias = bias + (1.0 - mask[:, None, None].float()) * torch.finfo(torch.float32).min
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, bias)
        states = T5Block.rms(x, self.norm)
        return states * mask[..., None]  # Cosmos zeroes the states of the padding tokens


# %% [patch-embedding] Patch embedding
class PatchEmbedding(nn.Module):
    """Stack the latent with the condition mask (1 on the given frames) and a padding mask (all 0 here), then cut
    1 × 2 × 2 patches and project each one's 72 numbers to d_model. No position is added: RoPE does that."""

    def __init__(self, c: Config):
        super().__init__()
        self.patch = c.patch
        self.proj = nn.Linear((c.latent_channels + 2) * math.prod(c.patch), c.d_model, bias=False)

    def forward(self, latent, condition_mask):  # (batch, 16, frames, h, w), (batch, 1, frames, h, w)
        x = torch.cat([latent, condition_mask, torch.zeros_like(condition_mask)], dim=1)
        b, ch, t, h, w = x.shape
        pt, ph, pw = self.patch
        x = x.reshape(b, ch, t // pt, pt, h // ph, ph, w // pw, pw).permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4)
        return self.proj(x).flatten(1, 3), (t // pt, h // ph, w // pw)  # (batch, frames · rows · cols, d_model)


# %% [rope] 3D RoPE
def rope(c: Config, grid, fps):
    """Of 128 dimensions per head, 44 turn with the frame (its time, frames / fps · 24), 42 with the row and 42 with
    the column; θ = 10000 scaled by NTK factors (rows and columns 3×). Halves rotate together (NeoX style)."""
    head = c.d_model // c.n_heads
    dims = {"h": head // 6 * 2, "w": head // 6 * 2}
    dims["t"] = head - dims["h"] - dims["w"]
    t, h, w = grid

    def angles(positions, dim, scale):
        theta = 10000.0 * scale ** (dim / (dim - 2))
        return torch.outer(positions, 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))

    at = angles(torch.arange(t, dtype=torch.float32) / fps * c.base_fps, dims["t"], c.rope_scale[0])
    ah = angles(torch.arange(h, dtype=torch.float32), dims["h"], c.rope_scale[1])
    aw = angles(torch.arange(w, dtype=torch.float32), dims["w"], c.rope_scale[2])
    grid_angles = torch.cat([
        at[:, None, None].expand(t, h, w, -1), ah[None, :, None].expand(t, h, w, -1), aw[None, None].expand(t, h, w, -1)
    ], dim=-1).flatten(0, 2)
    both = torch.cat([grid_angles, grid_angles], dim=-1)  # the same angles for both halves
    return both.cos(), both.sin()  # (tokens, 128)


def rotate(x, cos, sin):  # x: (batch, heads, tokens, 128)
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-second, first], dim=-1) * sin


# %% [timestep-embedding] Timestep embedding and adaLN-LoRA
class TimestepEmbedding(nn.Module):
    """Each latent frame's t → a sinusoid; an MLP turns it into temb (3 × d_model, added to every block's shift,
    scale and gate), and an RMSNorm of the sinusoid feeds each block's own small adaLN-LoRA."""

    def __init__(self, c: Config):
        super().__init__()
        self.d = c.d_model
        self.linear_1 = nn.Linear(c.d_model, c.d_model, bias=False)
        self.linear_2 = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.norm = nn.Parameter(torch.ones(c.d_model))

    def forward(self, t):  # t: (batch, frames), t = σ / (σ + 1)
        half = self.d // 2
        angles = t.flatten().float()[:, None] * torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        sinusoid = torch.cat([angles.cos(), angles.sin()], dim=-1)
        temb = self.linear_2(F.silu(self.linear_1(sinusoid)))
        embedded = sinusoid * torch.rsqrt(sinusoid.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm
        return temb.view(*t.shape, -1), embedded.view(*t.shape, -1)  # per frame


class AdaLNLoRA(nn.Module):
    """shift, scale, gate = Linear(256 → 3d)(Linear(d → 256)(SiLU(e))) + temb, per frame."""

    def __init__(self, c: Config, parts=3):
        super().__init__()
        self.parts = parts
        self.down = nn.Linear(c.d_model, c.adaln_lora, bias=False)
        self.up = nn.Linear(c.adaln_lora, parts * c.d_model, bias=False)

    def forward(self, x, embedded, temb):  # x: (batch, frames, tokens per frame, d); embedded, temb: (batch, frames, …)
        mod = self.up(self.down(F.silu(embedded))) + temb[..., : self.parts * x.shape[-1]]
        mod = mod[:, :, None].chunk(self.parts, dim=-1)
        x = F.layer_norm(x, x.shape[-1:], eps=1e-6) * (1 + mod[1]) + mod[0]
        return x, mod[2] if self.parts == 3 else None


# %% [attention] Self-attention
class SelfAttention(nn.Module):
    """Every token of every latent frame attends to every token of the clip: the given frames and the frames being
    predicted alike. q and k are RMS-normalised per head, then rotated by 3D RoPE."""

    def __init__(self, c: Config, kv_dim=None):
        super().__init__()
        self.heads = c.n_heads
        head = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(kv_dim or c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(kv_dim or c.d_model, c.d_model, bias=False)
        self.o = nn.Linear(c.d_model, c.d_model, bias=False)
        self.norm_q, self.norm_k = nn.Parameter(torch.ones(head)), nn.Parameter(torch.ones(head))

    def heads_of(self, x, proj, norm):
        y = proj(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        return y * torch.rsqrt(y.float().pow(2).mean(-1, keepdim=True) + 1e-5) * norm

    def forward(self, x, cos, sin, context=None):
        source = x if context is None else context
        q, k = self.heads_of(x, self.q, self.norm_q), self.heads_of(source, self.k, self.norm_k)
        v = self.v(source).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        if cos is not None:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))


# %% [cross-attention] Text cross-attention
class CrossAttention(SelfAttention):
    """Every token reads the prompt: queries from the token, keys and values from all 512 T5 states. The padding
    states are zero, so their keys are zero too and still take a share of the softmax (score 0)."""

    def __init__(self, c: Config):
        super().__init__(c, kv_dim=c.text_dim)

    def forward(self, x, text):
        return super().forward(x, None, None, context=text)


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


# %% [block] Cosmos block
class Block(nn.Module):
    """Self-attention, text cross-attention, MLP: each after its own adaLN-LoRA (shift, scale) and scaled by its
    own gate, all per latent frame, so the clean given frames and the noisy ones are modulated differently."""

    def __init__(self, c: Config):
        super().__init__()
        self.ada1, self.ada2, self.ada3 = AdaLNLoRA(c), AdaLNLoRA(c), AdaLNLoRA(c)
        self.attention = SelfAttention(c)
        self.cross = CrossAttention(c)
        self.mlp = MLP(c)

    def forward(self, x, embedded, temb, cos, sin, text):  # x: (batch, frames · s, d)
        b, n, d = x.shape
        f = embedded.shape[1]

        def framed(y):
            return y.view(b, f, n // f, d)

        h, gate = self.ada1(framed(x), embedded, temb)
        x = x + (gate * framed(self.attention(h.flatten(1, 2), cos, sin))).flatten(1, 2)
        h, gate = self.ada2(framed(x), embedded, temb)
        x = x + (gate * framed(self.cross(h.flatten(1, 2), text))).flatten(1, 2)
        h, gate = self.ada3(framed(x), embedded, temb)
        return x + (gate * framed(self.mlp(h.flatten(1, 2)))).flatten(1, 2)


# %% [output] Output layer and unpatchify
class OutputLayer(nn.Module):
    """adaLN-LoRA (shift and scale only), a Linear to 1 × 2 × 2 × 16 numbers per token, then the latent again."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.ada = AdaLNLoRA(c, parts=2)
        self.proj = nn.Linear(c.d_model, math.prod(c.patch) * c.latent_channels, bias=False)

    def forward(self, x, embedded, temb, grid):
        t, h, w = grid
        b, n, d = x.shape
        y, _ = self.ada(x.view(b, t, n // t, d), embedded, temb)
        y = self.proj(y.flatten(1, 2))  # (batch, tokens, 2 · 2 · 1 · 16): rows, columns, frames, channels
        pt, ph, pw = self.c.patch
        y = y.view(b, t, h, w, ph, pw, pt, -1).permute(0, 7, 1, 6, 2, 4, 3, 5)  # (batch, c, t, pt, h, ph, w, pw)
        return y.reshape(b, -1, t * pt, h * ph, w * pw)


# %% [scheduler-step] Karras σ, flow preconditioning and guidance
def karras_sigmas(steps=35, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """EDM's schedule: σ from 80 down to 0.002, evenly spaced in σ^(1/7). The last step's target is σ_min again."""
    ramp = torch.linspace(0, 1, steps, dtype=torch.float64)
    sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sigmas, sigmas[-1:]]).float()


def denoise(model_v, x, sigma, condition, keep):
    """The network is a rectified-flow model in t = σ / (σ + 1): its input is x scaled to (1 − t) x, and
    x̂₀ = (1 − t) x − t · v. The given frames (keep = 1) are replaced by their clean latents throughout."""
    t = sigma / (sigma + 1)
    x0 = (1 - t) * x - t * model_v((1 - t) * x * (1 - keep) + condition * keep, t)
    return keep * condition + (1 - keep) * x0


def euler_step(x, x0, sigma, sigma_next):
    """EDM's Euler step on dx/dσ = (x − x̂₀) / σ."""
    return x + (sigma_next - sigma) * (x - x0) / sigma


# %% [model] Full model and the Video2World loop
class Cosmos(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.timestep = TimestepEmbedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = OutputLayer(c)

    def forward(self, latent, t, condition_mask, text, fps):  # t: (batch, frames)
        x, grid = self.patchify(latent, condition_mask)
        cos, sin = rope(self.c, grid, fps)
        temb, embedded = self.timestep(t)
        for block in self.blocks:
            x = block(x, embedded, temb, cos, sin, text)
        return self.output(x, embedded, temb, grid)  # v for every latent pixel


@torch.no_grad()
def video2world(model, condition, given, text, negative_text, noise, steps=35, guidance=7.0, fps=16, sigma_given=1e-4):
    """condition: the VAE latents of the given frame(s) followed by copies of the last one, normalised;
    given: how many latent frames are given; noise: (batch, 16, frames, h, w) standard normal.
    Every step runs the model twice (prompt and negative prompt) and mixes x̂₀ with the guidance scale."""
    b, _, frames, h, w = noise.shape
    keep = torch.zeros(1, 1, frames, 1, 1)
    keep[:, :, :given] = 1
    mask = keep.expand(b, 1, frames, h, w)
    t_given = torch.tensor(sigma_given) / (sigma_given + 1)
    sigmas = karras_sigmas(steps)
    x = noise * sigmas[0]
    for i in range(steps):
        sigma = sigmas[i]

        def model_v(inputs, t, words):
            per_frame = (keep * t_given + (1 - keep) * t).view(1, frames).expand(b, frames)
            return model(inputs, per_frame, mask, words, fps)

        x0 = denoise(lambda inputs, t: model_v(inputs, t, text), x, sigma, condition, keep)
        x0_negative = denoise(lambda inputs, t: model_v(inputs, t, negative_text), x, sigma, condition, keep)
        x = euler_step(x, x0 + guidance * (x0 - x0_negative), sigma, sigmas[i + 1])
    return x  # the VAE decodes x · std + mean
`;export{e as default};
