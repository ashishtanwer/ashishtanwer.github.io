const e=`# TRELLIS (microsoft/TRELLIS-image-large), the image-to-3D model in the TRELLIS tab, written as plain PyTorch blocks.
# DINOv2 reads the photo; a rectified-flow transformer denoises a 16³ latent that a 3D conv decoder turns into the
# occupied voxels of a 64³ grid (the sparse structure); a second flow transformer denoises 8 numbers on every occupied
# voxel (the structured latent); a sparse transformer decodes each voxel's 8 numbers into 32 3D Gaussians.
# TRELLIS's own code runs the sparse parts with CUDA libraries (spconv, flash-attn or xformers) and renders with
# mip-splatting's CUDA rasterizer; here they are plain PyTorch: a submanifold convolution as 27 gathered matrix
# products, windowed attention as one attention per window, and the rasterizer's per-pixel rules.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    width: int = 1024  # both flow transformers
    n_heads: int = 16  # 64 dimensions per head
    n_layers: int = 24
    cond_width: int = 1024  # DINOv2 ViT-L/14's tokens
    grid: int = 16  # stage 1 denoises an 8 × 16³ latent
    channels: int = 8  # latent channels in both stages
    decoder_channels: tuple = (512, 128, 32)  # the structure decoder at 16³, 32³, 64³
    resolution: int = 64  # the sparse structure's grid
    io_channels: int = 128  # stage 2's convolutions at 64³, before it pools to 32³
    gs_width: int = 768  # the Gaussian decoder
    gs_heads: int = 12
    gs_layers: int = 12
    window: int = 8  # its attention windows: 8³ voxels, shifted by 4 in every other block
    gaussians: int = 32  # per voxel
    voxel_size: float = 1.5  # how far (in voxels) a Gaussian may sit from its voxel's centre
    steps: int = 25
    cfg: float = 5.0  # guidance (1 + 5)·v(image) − 5·v(∅) …
    cfg_interval: tuple = (0.5, 1.0)  # … only while t is in this range
    rescale_t: float = 3.0
    sigma_min: float = 1e-5


# %% [image-encoder] DINOv2 image tokens
def image_tokens(dinov2, image):
    """image: (batch, 3, 518, 518) in [0, 1], the object on black. ImageNet normalisation, DINOv2 ViT-L/14 with 4
    registers; its last block's output before the final norm, layer-normalised without a scale: 1 + 4 + 37² = 1374
    tokens of 1024. The unconditional input is the same shape, all zeros."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    hidden = dinov2((image - mean) / std, output_hidden_states=True).hidden_states[-1]
    return F.layer_norm(hidden, hidden.shape[-1:])


# %% [timestep-embedding] Timestep embedding
class TimestepEmbedder(nn.Module):
    def __init__(self, width, frequencies=256):
        super().__init__()
        self.frequencies = frequencies
        self.mlp = nn.Sequential(nn.Linear(frequencies, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):  # t: (batch,) = 1000 · flow time
        half = self.frequencies // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        angles = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))


# %% [position] 3D position embedding
def position_embedding(coords, width):
    """sin then cos of each of x, y, z at width // 6 frequencies 10000^(−i/(width/6)), zero-padded to width."""
    n_freq = width // 3 // 2
    freqs = 1.0 / 10000 ** (torch.arange(n_freq, dtype=torch.float32) / n_freq)
    angles = coords.float().reshape(-1)[:, None] * freqs[None]  # (n · 3, n_freq)
    embed = torch.cat([angles.sin(), angles.cos()], dim=-1).reshape(len(coords), -1)
    return F.pad(embed, (0, width - embed.shape[1]))


# %% [attention] Self-attention with QK RMS-norm
class RMSNorm(nn.Module):
    """Per head: q / ‖q‖ · √64 · γ (a learned γ per head and dimension)."""

    def __init__(self, heads, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x):  # (…, heads, dim)
        return F.normalize(x, dim=-1) * self.gamma * x.shape[-1] ** 0.5


class SelfAttention(nn.Module):
    def __init__(self, width, heads, qk_norm=True):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(heads, width // heads) if qk_norm else None
        self.k_norm = RMSNorm(heads, width // heads) if qk_norm else None
        self.to_out = nn.Linear(width, width)

    def qkv(self, x):  # (…, tokens, width) → q, k, v: (…, tokens, heads, 64)
        q, k, v = self.to_qkv(x).unflatten(-1, (3, self.heads, -1)).unbind(-3)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    def forward(self, x):  # (batch, tokens, width): every token sees every token
        q, k, v = self.qkv(x)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.to_out(out.transpose(1, 2).flatten(2))


# %% [cross-attention] Cross-attention to the image tokens
class CrossAttention(nn.Module):
    def __init__(self, width, cond_width, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(width, width)
        self.to_kv = nn.Linear(cond_width, 2 * width)
        self.to_out = nn.Linear(width, width)

    def weights(self, x, cond):  # (batch, heads, tokens, 1374): where each token looks in the image
        q = self.to_q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k = self.to_kv(cond).unflatten(-1, (2, self.heads, -1))[..., 0, :, :].transpose(1, 2)
        return torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)

    def forward(self, x, cond):
        q = self.to_q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        k, v = self.to_kv(cond).unflatten(-1, (2, self.heads, -1)).permute(2, 0, 3, 1, 4)  # 2 × (batch, heads, 1374, 64)
        return self.to_out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate='tanh'))


# %% [block] Flow transformer block
class Block(nn.Module):
    """adaLN from the timestep (shift, scale and gate for attention and MLP), then self-attention with QK RMS-norm,
    cross-attention to the 1374 image tokens (a plain LayerNorm, no timestep), and the MLP."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.width, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(c.width, c.n_heads)
        self.norm2 = nn.LayerNorm(c.width, eps=1e-6)
        self.cross_attn = CrossAttention(c.width, c.cond_width, c.n_heads)
        self.norm3 = nn.LayerNorm(c.width, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(c.width)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(c.width, 6 * c.width))

    def forward(self, x, temb, cond):  # x: (batch, tokens, width); temb: (batch, width)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(temb)[:, None].chunk(6, dim=-1)
        x = x + gate1 * self.self_attn(self.norm1(x) * (1 + scale1) + shift1)
        x = x + self.cross_attn(self.norm2(x), cond)
        return x + gate2 * self.mlp(self.norm3(x) * (1 + scale2) + shift2)


# %% [flow] Rectified flow and the Euler sampler
# Training puts x_t = (1 − t)·x₀ + (σ_min + (1 − σ_min)·t)·ε on a near-straight line from data (t = 0) to noise (t = 1)
# and learns its velocity; sampling walks back in 25 Euler steps on times bunched towards t = 1.
def flow_times(steps=25, rescale=3.0):
    t = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return rescale * t / (1 + (rescale - 1) * t)  # (steps + 1,) from 1 to 0


def guided_velocity(model, x, t, cond, neg_cond, c: Config):
    """v(x, t | image), pushed away from v(x, t | zeros) with strength 5, but only for t in [0.5, 1]."""
    tt = torch.tensor([1000 * t], dtype=torch.float32)
    v = model(x, tt, cond)
    if c.cfg_interval[0] <= t <= c.cfg_interval[1]:
        return (1 + c.cfg) * v - c.cfg * model(x, tt, neg_cond)
    return v


def predicted_x0(x, t, v, sigma_min=1e-5):
    return (1 - sigma_min) * x - (sigma_min + (1 - sigma_min) * t) * v


def euler_step(x, v, t, t_next):
    return x - (t - t_next) * v


# %% [structure-flow] Stage 1: the sparse-structure flow transformer
class StructureFlow(nn.Module):
    """Dense: every one of the 16³ = 4096 cells is a token (patch size 1) with a fixed 3D position."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.t_embedder = TimestepEmbedder(c.width)
        self.input_layer = nn.Linear(c.channels, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.out_layer = nn.Linear(c.width, c.channels)
        cells = torch.stack(torch.meshgrid(*[torch.arange(c.grid)] * 3, indexing='ij'), -1).reshape(-1, 3)
        self.register_buffer('pos_emb', position_embedding(cells, c.width), persistent=False)

    def forward(self, x, t, cond):  # x: (batch, 8, 16, 16, 16) → velocity, same shape
        h = self.input_layer(x.flatten(2).transpose(1, 2)) + self.pos_emb  # (batch, 4096, 1024)
        temb = self.t_embedder(t)
        for block in self.blocks:
            h = block(h, temb, cond)
        h = self.out_layer(F.layer_norm(h, h.shape[-1:]))
        return h.transpose(1, 2).reshape(x.shape)


# %% [structure-decoder] The sparse structure: a 3D conv decoder
class ChannelNorm(nn.LayerNorm):
    def forward(self, x):  # LayerNorm over channels of (batch, C, D, H, W)
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)


class ResBlock3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(channels), ChannelNorm(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))


class Upsample3d(nn.Module):
    """A conv to 8× the channels, then each voxel's 8 groups become its 2 × 2 × 2 children (3D pixel shuffle)."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, out_channels * 8, 3, padding=1)

    def forward(self, x):
        b, _, d, h, w = x.shape
        y = self.conv(x).view(b, -1, 2, 2, 2, d, h, w)
        return y.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, -1, 2 * d, 2 * h, 2 * w)


class StructureDecoder(nn.Module):
    """8 × 16³ → 512 channels → two ResBlocks at each of 16³, 32³ and 64³ (512, 128, 32 channels) → one logit per
    voxel of the 64³ grid. A voxel is occupied where the logit is above 0."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        chs = c.decoder_channels
        self.input_layer = nn.Conv3d(c.channels, chs[0], 3, padding=1)
        self.middle_block = nn.Sequential(ResBlock3d(chs[0]), ResBlock3d(chs[0]))
        layers = []
        for i, ch in enumerate(chs):
            layers += [ResBlock3d(ch), ResBlock3d(ch)]
            if i < len(chs) - 1:
                layers.append(Upsample3d(ch, chs[i + 1]))
        self.blocks = nn.Sequential(*layers)
        self.out_layer = nn.Sequential(ChannelNorm(chs[-1]), nn.SiLU(), nn.Conv3d(chs[-1], 1, 3, padding=1))

    def forward(self, z):  # (batch, 8, 16, 16, 16) → (batch, 1, 64, 64, 64) logits
        return self.out_layer(self.blocks(self.middle_block(self.input_layer(z))))


def occupied(logits):
    """The occupied voxels' (x, y, z) in lexicographic order, as TRELLIS's argwhere gives them."""
    return torch.argwhere(logits[0, 0] > 0)


# %% [sparse-conv] Submanifold sparse convolution
class SparseConv3d(nn.Module):
    """3 × 3 × 3 convolution evaluated only at the occupied voxels, reading only occupied neighbours: the same as a
    dense conv3d on a grid of zeros, read at the occupied voxels. The weight is spconv's (out, 3, 3, 3, in)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, feats, coords, size):  # feats: (n, in); coords: (n, 3) in [0, size)
        index = torch.full((size,) * 3, -1, dtype=torch.long)
        index[coords.unbind(-1)] = torch.arange(len(coords))
        out = self.bias.expand(len(coords), -1).clone()
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    neighbour = coords + torch.tensor([dx - 1, dy - 1, dz - 1])
                    inside = ((neighbour >= 0) & (neighbour < size)).all(-1)
                    j = torch.full((len(coords),), -1, dtype=torch.long)
                    j[inside] = index[neighbour[inside].unbind(-1)]
                    hit = j >= 0
                    out[hit] += feats[j[hit]] @ self.weight[:, dx, dy, dz].T
        return out


# %% [sparse-resblock] Sparse ResBlock, pooling and unpooling
def pool(feats, coords):
    """Pool the occupied children of each 2³ block: (n, C) at 64³ → (m, C) at 32³, and each child's parent. TRELLIS
    takes a scatter "mean" onto zeros that counts the zero too, so a parent gets sum(children) / (children + 1)."""
    parents, child_to_parent = torch.unique(coords // 2, dim=0, return_inverse=True)
    total = torch.zeros(len(parents), feats.shape[1]).index_add_(0, child_to_parent, feats)
    children = torch.bincount(child_to_parent, minlength=len(parents))
    return total / (children[:, None] + 1), parents, child_to_parent


class SparseResBlock(nn.Module):
    """norm → SiLU → conv → norm, scaled and shifted by the timestep → SiLU → conv, plus a skip. \`resample\` pools
    (down) or copies each parent to its children (up) before everything else."""

    def __init__(self, channels, out_channels, emb_width):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.conv1 = SparseConv3d(channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)
        self.conv2 = SparseConv3d(out_channels, out_channels)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_width, 2 * out_channels))
        self.skip = nn.Linear(channels, out_channels) if channels != out_channels else nn.Identity()

    def forward(self, x, coords, size, temb):
        scale, shift = self.emb(temb).chunk(2, dim=-1)
        h = self.conv1(F.silu(self.norm1(x)), coords, size)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift), coords, size)
        return h + self.skip(x)


# %% [latent-flow] Stage 2: the structured-latent flow transformer
class LatentFlow(nn.Module):
    """8 numbers on each of the n occupied voxels. Sparse ResBlocks at 64³ (128 channels), pooled to 32³ (m parents,
    1024 channels), 24 transformer blocks over those m tokens, then back up with skips, to a velocity per voxel."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.t_embedder = TimestepEmbedder(c.width)
        self.input_layer = nn.Linear(c.channels, c.io_channels)
        self.input_blocks = nn.ModuleList(
            [SparseResBlock(c.io_channels, c.io_channels, c.width), SparseResBlock(c.io_channels, c.width, c.width)]
        )
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.out_blocks = nn.ModuleList(
            [SparseResBlock(2 * c.width, c.io_channels, c.width), SparseResBlock(2 * c.io_channels, c.io_channels, c.width)]
        )
        self.out_layer = nn.Linear(c.io_channels, c.channels)

    def forward(self, feats, coords, t, cond):  # feats: (n, 8); coords: (n, 3) at 64³; t: (1,); cond: (1, 1374, 1024)
        r = self.c.resolution
        temb = self.t_embedder(t)
        h0 = self.input_blocks[0](self.input_layer(feats), coords, r, temb)  # (n, 128) at 64³
        pooled, parents, child_to_parent = pool(h0, coords)
        h1 = self.input_blocks[1](pooled, parents, r // 2, temb)  # (m, 1024) at 32³
        h = (h1 + position_embedding(parents, self.c.width))[None]
        for block in self.blocks:
            h = block(h, temb, cond)
        h = torch.cat([h[0], h1], dim=-1)[child_to_parent]  # unpool: every child gets its parent's 2048 numbers
        h = self.out_blocks[0](h, coords, r, temb)
        h = self.out_blocks[1](torch.cat([h, h0], dim=-1), coords, r, temb)
        return self.out_layer(F.layer_norm(h, h.shape[-1:]))


# %% [window-attention] Shifted-window attention over voxels
def windows(coords, window=8, shift=0):
    """Voxels grouped by the 8³ window they fall in (after shifting every coordinate by \`shift\`): a list of index
    tensors, one per occupied window."""
    cell = (coords + shift) // window
    order = torch.argsort(cell[:, 0] * 4096 + cell[:, 1] * 64 + cell[:, 2], stable=True)
    _, counts = torch.unique_consecutive(cell[order], dim=0, return_counts=True)
    return order.split(counts.tolist())


class WindowBlock(nn.Module):
    """LayerNorm → attention inside each window → add; LayerNorm → MLP → add. Odd blocks shift the windows by half a
    window so neighbouring windows exchange information."""

    def __init__(self, width, heads, shift):
        super().__init__()
        self.shift = shift
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(width, heads, qk_norm=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(width)

    def forward(self, x, coords, window):  # x: (n, width)
        q, k, v = self.attn.qkv(self.norm1(x))
        mixed = torch.empty_like(q)
        for idx in windows(coords, window, self.shift):
            mixed[idx] = F.scaled_dot_product_attention(
                q[idx].transpose(0, 1), k[idx].transpose(0, 1), v[idx].transpose(0, 1)
            ).transpose(0, 1)
        x = x + self.attn.to_out(mixed.flatten(1))
        return x + self.mlp(self.norm2(x))


# %% [gaussian-decoder] Structured latent → 3D Gaussians
def hammersley_offsets(count, voxel_size):
    """The 32 Gaussians' fixed starting spots in a voxel, a Hammersley set in [−1, 1]³, as pre-tanh offsets."""

    def radical_inverse(base, n):
        value, inv = 0.0, 1.0 / base
        while n > 0:
            value += (n % base) * inv
            n //= base
            inv /= base
        return value

    points = torch.tensor([[i / count, radical_inverse(2, i), radical_inverse(3, i)] for i in range(count)])
    return torch.atanh((points * 2 - 1) / voxel_size)


class GaussianDecoder(nn.Module):
    """Each voxel's 8 numbers (de-normalised) → Linear to 768 + its 3D position → 12 window blocks → LayerNorm →
    Linear to 448 = 32 Gaussians × (offset 3, colour 3, scale 3, rotation 4, opacity 1)."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.input_layer = nn.Linear(c.channels, c.gs_width)
        shifts = [c.window // 2 * (i % 2) for i in range(c.gs_layers)]
        self.blocks = nn.ModuleList(WindowBlock(c.gs_width, c.gs_heads, s) for s in shifts)
        self.out_layer = nn.Linear(c.gs_width, 14 * c.gaussians)
        self.register_buffer('offset_perturbation', hammersley_offsets(c.gaussians, c.voxel_size))

    def forward(self, slat, coords):  # (n, 8), (n, 3) → (n, 448)
        h = self.input_layer(slat) + position_embedding(coords, self.c.gs_width)
        for block in self.blocks:
            h = block(h, coords, self.c.window)
        return self.out_layer(F.layer_norm(h, h.shape[-1:]))

    def gaussians(self, out, coords):
        """Activated parameters, as TRELLIS's Gaussian class reads them, in a [−0.5, 0.5]³ box."""
        c, g = self.c, self.c.gaussians
        offset, colour, scale, rotation, opacity = out.split([3 * g, 3 * g, 3 * g, 4 * g, g], dim=-1)
        offset = torch.tanh(offset.view(-1, g, 3) + self.offset_perturbation) / c.resolution * 0.5 * c.voxel_size
        xyz = ((coords.float() + 0.5) / c.resolution)[:, None] + offset - 0.5
        scale_bias = math.log(math.expm1(0.004))  # softplus⁻¹(0.004)
        scale = (F.softplus(scale.reshape(-1, 3) + scale_bias) ** 2 + 9e-4**2).sqrt()  # with the 3D filter
        rotation = F.normalize(0.1 * rotation.reshape(-1, 4) + torch.tensor([1.0, 0, 0, 0]), dim=-1)
        return {
            'xyz': xyz.reshape(-1, 3),
            'colour': (0.28209479177387814 * colour.reshape(-1, 3) + 0.5).clamp(min=0),  # SH degree 0
            'scale': scale,
            'rotation': rotation,
            'opacity': torch.sigmoid(opacity.reshape(-1, 1) + math.log(0.1 / 0.9)),
        }


# %% [splat] Rendering the Gaussians
def look_at(yaw, pitch, radius=2.0):
    """TRELLIS's orbit camera (utils3d's OpenCV look-at, z up): world → camera rotation and translation."""
    eye = radius * torch.tensor([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    z = -eye
    x = torch.linalg.cross(-torch.tensor([0.0, 0, 1]), z)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x / x.norm(), y / y.norm(), z / z.norm()])
    return R, -R @ eye


def splat(g, R, T, size=128, fov=math.radians(40), background=(1.0, 1.0, 1.0), kernel=0.1, reach=3.5):
    """mip-splatting's rasterizer rules, per pixel: project each Gaussian (EWA), widen it by the 2D filter \`kernel\`
    and scale its opacity by √(det Σ / det(Σ + kernel·I)), sort front to back, α = min(0.99, o·exp(−½ dᵀΣ⁻¹d)), skip
    α < 1/255, stop before the light left falls under 10⁻⁴. Like the CUDA kernel, a Gaussian reaches the pixels of
    the 16 × 16 tiles its 3σ square touches; past 3.5σ its α is always under 1/255, so only that far is searched."""
    p = g['xyz'] @ R.T + T
    keep = p[:, 2] > 0.2
    p, colour, opacity = p[keep], g['colour'][keep], g['opacity'][keep, 0]
    w, x, y, z = g['rotation'][keep].unbind(-1)
    rot = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).view(-1, 3, 3)  # fmt: skip
    M = rot * g['scale'][keep][:, None, :]
    cov3 = R @ (M @ M.transpose(1, 2)) @ R.T  # camera-space covariance
    tan = math.tan(fov / 2)
    focal = size / (2 * tan)
    tx = (p[:, 0] / p[:, 2]).clamp(-1.3 * tan, 1.3 * tan) * p[:, 2]
    ty = (p[:, 1] / p[:, 2]).clamp(-1.3 * tan, 1.3 * tan) * p[:, 2]
    J = torch.zeros(len(p), 2, 3)
    J[:, 0, 0] = focal / p[:, 2]
    J[:, 0, 2] = -focal * tx / p[:, 2] ** 2
    J[:, 1, 1] = focal / p[:, 2]
    J[:, 1, 2] = -focal * ty / p[:, 2] ** 2
    cov2 = J @ cov3 @ J.transpose(1, 2)
    a, b, d = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det0 = (a * d - b * b).clamp(min=1e-6)
    det1 = ((a + kernel) * (d + kernel) - b * b).clamp(min=1e-6)
    coef = torch.where((det0 > 1e-6) & (det1 > 1e-6), (det0 / (det1 + 1e-6) + 1e-6).sqrt(), torch.zeros_like(det0))
    a, d = a + kernel, d + kernel
    det = a * d - b * b
    conic = torch.stack([d / det, -b / det, a / det], -1)
    mid = (a + d) / 2
    spread = (mid * mid - det).clamp(min=0.1).sqrt()
    sigma = (mid + spread).sqrt()
    centre = torch.stack([focal * p[:, 0] / p[:, 2] + size / 2 - 0.5, focal * p[:, 1] / p[:, 2] + size / 2 - 0.5], -1)
    alpha_max = opacity * coef
    # every (Gaussian, pixel) pair within reach, ranked front to back
    order = torch.argsort(p[:, 2])
    depth_rank = torch.empty_like(order)
    depth_rank[order] = torch.arange(len(order))
    tiles = (size + 15) // 16
    three = torch.ceil(3 * sigma)  # the kernel's radius
    tile_lo = ((centre - three[:, None]) / 16).long().clamp(0, tiles)  # its tile rectangle [lo, hi)
    tile_hi = ((centre + three[:, None] + 15) / 16).long().clamp(0, tiles)
    radius = torch.ceil(reach * sigma + 0.5).long().clamp(max=size)
    ranks, pix, alphas, gids = [], [], [], []
    groups = [order[radius[order] == r] for r in radius[order].unique().tolist()]
    for ids in (chunk for group in groups for chunk in group.split(65536)):  # bounded memory per pass
        r = int(radius[ids[0]])
        off = torch.arange(-r, r + 1)
        oy, ox = torch.meshgrid(off, off, indexing='ij')
        px = centre[ids, 0, None].round().long() + ox.reshape(1, -1)
        py = centre[ids, 1, None].round().long() + oy.reshape(1, -1)
        dx, dy = centre[ids, 0, None] - px, centre[ids, 1, None] - py
        k = conic[ids]
        power = -0.5 * (k[:, 0, None] * dx * dx + k[:, 2, None] * dy * dy) - k[:, 1, None] * dx * dy
        alpha = (alpha_max[ids, None] * power.exp()).clamp(max=0.99)
        in_tiles = (
            (px // 16 >= tile_lo[ids, 0, None]) & (px // 16 < tile_hi[ids, 0, None])
            & (py // 16 >= tile_lo[ids, 1, None]) & (py // 16 < tile_hi[ids, 1, None])
        )  # fmt: skip
        ok = in_tiles & (power <= 0) & (alpha >= 1 / 255) & (px >= 0) & (px < size) & (py >= 0) & (py < size)
        ranks.append(depth_rank[ids, None].expand_as(px)[ok])
        pix.append((py * size + px)[ok])
        alphas.append(alpha[ok])
        gids.append(ids[:, None].expand_as(px)[ok])
    rank, pixel, alpha, gid = torch.cat(ranks), torch.cat(pix), torch.cat(alphas), torch.cat(gids)
    sort = torch.argsort(pixel * (len(p) + 1) + rank)
    pixel, alpha, gid = pixel[sort], alpha[sort].double(), gid[sort]
    # light left in front of each pair, per pixel: exclusive cumulative product of (1 − α) within the pixel
    log_left = torch.log1p(-alpha)
    cum = log_left.cumsum(0)
    start = torch.searchsorted(pixel, pixel)  # first pair of each pixel
    before = torch.exp(cum - log_left - torch.where(start > 0, cum[(start - 1).clamp(min=0)], torch.zeros_like(cum)))
    after = before * (1 - alpha)
    used = after >= 1e-4  # T falls monotonically, so this drops the pair that would cross 10⁻⁴ and all behind it
    image = torch.zeros(size * size, 3, dtype=torch.float64)
    image.index_add_(0, pixel[used], colour[gid[used]].double() * (alpha * before)[used, None])
    left = torch.ones(size * size, dtype=torch.float64).scatter_reduce_(0, pixel[used], after[used], 'amin')
    return (image + left[:, None] * torch.tensor(background, dtype=torch.float64)).float().view(size, size, 3)


# %% [step] One denoising step
def denoise_step(model, x, t, t_next, cond, neg_cond, c: Config = Config()):
    """One Euler step of either stage: the guided velocity, the x̂₀ it points at, and x at the next time."""
    v = guided_velocity(model, x, t, cond, neg_cond, c)
    return euler_step(x, v, t, t_next), predicted_x0(x, t, v, c.sigma_min), v


# %% [model] Image to 3D
@torch.no_grad()
def image_to_3d(dinov2, structure_flow, structure_decoder, latent_flow, decoder, image, mean, std, seed=42,
                c: Config = Config()):  # fmt: skip
    """TRELLIS's run(): image tokens (and zeros for guidance) → stage 1 from noise → occupied voxels → stage 2 from
    noise on them → de-normalised structured latent → Gaussians."""
    cond = image_tokens(dinov2, image)
    neg = torch.zeros_like(cond)
    torch.manual_seed(seed)
    times = flow_times(c.steps, c.rescale_t).tolist()
    x = torch.randn(1, c.channels, c.grid, c.grid, c.grid)
    for t, t_next in zip(times[:-1], times[1:]):
        x, _, _ = denoise_step(structure_flow, x, t, t_next, cond, neg, c)
    coords = occupied(structure_decoder(x))
    feats = torch.randn(len(coords), c.channels)
    model = lambda f, tt, cc: latent_flow(f, coords, tt, cc)  # noqa: E731
    for t, t_next in zip(times[:-1], times[1:]):
        feats, _, _ = denoise_step(model, feats, t, t_next, cond, neg, c)
    slat = feats * std + mean
    return decoder.gaussians(decoder(slat, coords), coords), coords, slat
`;export{e as default};
