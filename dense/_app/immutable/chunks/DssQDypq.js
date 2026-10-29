const e=`# Matrix-Game 2.0 (Skywork/Matrix-Game-2.0, the universal distilled model), the interactive world model in the
# Matrix-Game tab, written as plain PyTorch blocks. It is Wan 2.1's 1.3B video DiT without the text branch, with an
# action module (keyboard and mouse) in its first 15 blocks and a KV cache, so it makes video chunk by chunk:
# 3 latent frames (12 video frames) at a time, each chunk in 3 flow-matching steps.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    latent_channels: int = 16  # Wan 2.1 VAE: one latent pixel per 8 × 8 pixels and 4 video frames
    condition_channels: int = 20  # first-frame mask (4) + image latent (16)
    d_model: int = 1536
    n_heads: int = 12  # 128 dimensions per head
    n_layers: int = 30
    d_mlp: int = 8960
    frequency_dim: int = 256
    image_dim: int = 1280  # CLIP ViT-H/14 features of the first frame, 257 tokens
    eps: float = 1e-6
    frames_per_chunk: int = 3
    window_frames: int = 6  # attention sees the last 6 latent frames, this chunk's 3 included
    action_blocks: int = 15  # blocks 0–14 carry the action module
    keyboard_dim: int = 4  # W, S, A, D
    mouse_dim: int = 2  # camera pitch, yaw
    keyboard_embed: int = 128
    action_dim: int = 1024  # mouse and keyboard attention: 16 heads × 64
    action_heads: int = 16
    video_frames_per_latent: int = 4
    action_window: int = 3  # latent frame f reads the actions of video frames 4f − 12 … 4f − 1
    action_rope_dims: int = 8  # of 64 dimensions per action head, only the first 8 turn with time
    action_rope_theta: float = 256.0


# %% [patch-embedding] Patch embedding
def first_frame_condition(image_latent):
    """The given image is latent frame 0: a 4-channel mask (1 on frame 0, 0 after) and the VAE latent of the image
    followed by gray frames. image_latent: (batch, 16, frames, 44, 80) → (batch, 20, frames, 44, 80)."""
    mask = torch.zeros_like(image_latent[:, :4])
    mask[:, :, 0] = 1
    return torch.cat([mask, image_latent], dim=1)


class PatchEmbedding(nn.Module):
    """Stack the noisy latent on the condition (36 channels), cut 1 × 2 × 2 patches, project each to d_model."""

    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Conv3d(c.latent_channels + c.condition_channels, c.d_model, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, noisy, condition):  # (batch, 16, frames, 44, 80), (batch, 20, frames, 44, 80)
        x = self.proj(torch.cat([noisy, condition], dim=1))  # (batch, d_model, frames, 22, 40)
        return x.flatten(2).transpose(1, 2), tuple(x.shape[2:])  # 880 tokens per frame, frame after frame


# %% [timestep-embedding] Timestep embedding
def sinusoid(t, dim):
    """cos then sin of t at dim/2 frequencies 10000^(−i/(dim/2))."""
    half = dim // 2
    angles = t.double()[:, None] * 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    return torch.cat([angles.cos(), angles.sin()], dim=1)


class TimestepEmbedding(nn.Module):
    """Each latent frame has its own t, so e and the six modulation vectors are per frame."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.mlp_in = nn.Linear(c.frequency_dim, c.d_model)
        self.mlp_out = nn.Linear(c.d_model, c.d_model)
        self.project = nn.Linear(c.d_model, 6 * c.d_model)

    def forward(self, t):  # t: (batch, frames)
        b, f = t.shape
        e = self.mlp_out(F.silu(self.mlp_in(sinusoid(t.flatten(), self.frequency_dim).float())))
        return e.view(b, f, 1, -1), self.project(F.silu(e)).view(b, f, 6, -1)  # e for the head; shift/scale/gate × 2


# %% [image-conditioning] First-frame CLIP tokens
class ImageProjection(nn.Module):
    """The first frame's 257 CLIP ViT-H/14 tokens (from 31 of its 32 blocks) → the keys and values of every
    block's cross-attention. Matrix-Game has no text: this image is the only context besides the actions."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm_in = nn.LayerNorm(c.image_dim)
        self.fc1 = nn.Linear(c.image_dim, c.image_dim)
        self.fc2 = nn.Linear(c.image_dim, c.d_model)
        self.norm_out = nn.LayerNorm(c.d_model)

    def forward(self, clip_tokens):  # (batch, 257, 1280)
        return self.norm_out(self.fc2(F.gelu(self.fc1(self.norm_in(clip_tokens)))))


# %% [rope] 3D RoPE
def rope_angles(c: Config, frames, rows, cols, start_frame):
    """Of a head's 64 complex pairs, 22 turn with the latent frame index, 21 with the row, 21 with the column.
    The frame index counts from the start of the video, so cached frames keep their place in time."""
    d = c.d_model // c.n_heads
    sizes = (d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6))  # 44, 42, 42 dimensions

    def angles(count, dim, start=0):
        return torch.outer(torch.arange(start, start + count, dtype=torch.float64), 10000.0 ** -(torch.arange(0, dim, 2, dtype=torch.float64) / dim))

    grid = torch.cat([
        angles(frames, sizes[0], start_frame)[:, None, None].expand(frames, rows, cols, -1),
        angles(rows, sizes[1])[None, :, None].expand(frames, rows, cols, -1),
        angles(cols, sizes[2])[None, None, :].expand(frames, rows, cols, -1),
    ], dim=-1)
    return grid.reshape(frames * rows * cols, 1, d // 2)  # (tokens, 1, 64)


def rotate(x, angles):
    """x: (batch, tokens, heads, d). Adjacent pairs of coordinates are complex numbers turned by their angle."""
    pairs = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * torch.polar(torch.ones_like(angles), angles)).flatten(3).type_as(x)


# %% [attention] Self-attention over the cached frames
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def with_cache(cache, k, v, window):
    """Keys and values of the last \`window\` latent frames (dim 1): the cache's clean frames, then this chunk's.
    This chunk's k, v wait in cache['new'] until the clean pass at t = 0 commits them (see \`commit\`)."""
    cache["new"] = (k, v)
    if "k" in cache:
        k, v = torch.cat([cache["k"], k], dim=1)[:, -window:], torch.cat([cache["v"], v], dim=1)[:, -window:]
    return k, v


class SelfAttention(nn.Module):
    """Every token of the chunk attends to every token of the last 6 latent frames: the previous chunk's 3 clean
    frames from the KV cache, and this chunk's 3 frames. Nothing earlier, nothing later."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads, self.window = c.n_heads, c.window_frames
        self.q, self.k, self.v, self.o = (nn.Linear(c.d_model, c.d_model) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(c.d_model, c.eps), RMSNorm(c.d_model, c.eps)

    def forward(self, x, angles, frames, cache):  # x: (batch, frames × 880, d_model)
        b, n, d = x.shape
        q = rotate(self.norm_q(self.q(x)).view(b, n, self.n_heads, -1), angles)
        k = rotate(self.norm_k(self.k(x)).view(b, n, self.n_heads, -1), angles)
        v = self.v(x).view(b, n, self.n_heads, -1)
        per_frame = (b, frames, n // frames, self.n_heads, -1)
        keys, values = with_cache(cache, k.view(per_frame), v.view(per_frame), self.window)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), keys.flatten(1, 2).transpose(1, 2), values.flatten(1, 2).transpose(1, 2))
        return self.o(out.transpose(1, 2).reshape(b, n, d))


# %% [cross-attention] Image cross-attention
class ImageCrossAttention(nn.Module):
    """Every token reads the 257 projected CLIP tokens of the first frame. No RoPE, no mask, no cache needed:
    the keys and values are the same for every chunk."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.q, self.k, self.v, self.o = (nn.Linear(c.d_model, c.d_model) for _ in range(4))
        self.norm_q, self.norm_k = RMSNorm(c.d_model, c.eps), RMSNorm(c.d_model, c.eps)

    def forward(self, x, image_tokens):  # image_tokens: (batch, 257, d_model), from ImageProjection
        b, h = x.shape[0], self.n_heads
        q = self.norm_q(self.q(self.norm(x))).view(b, -1, h, x.shape[-1] // h).transpose(1, 2)
        k = self.norm_k(self.k(image_tokens)).view(b, -1, h, x.shape[-1] // h).transpose(1, 2)
        v = self.v(image_tokens).view(b, -1, h, x.shape[-1] // h).transpose(1, 2)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))


# %% [mouse] Mouse attention
def rms(x, eps=1e-6):
    """RMSNorm without a weight: the checkpoint has q/k-norm weights here that the reference code never applies."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def action_rope(c: Config, x, start_frame):
    """x: (rows, frames, heads, 64). Turn the first 8 coordinates by the latent frame index (θ = 256), as complex
    pairs; the other 56 stay as they are."""
    n = c.action_rope_dims
    position = torch.arange(start_frame, start_frame + x.shape[1], dtype=torch.float32)
    angles = torch.outer(position, c.action_rope_theta ** -(torch.arange(0, n, 2).float() / n)).repeat_interleave(2, dim=1)
    cos, sin = angles.cos()[None, :, None], angles.sin()[None, :, None]  # (1, frames, 1, 8)
    head = x[..., :n].float()
    pairs = head.reshape(*head.shape[:-1], -1, 2)
    turned = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
    return torch.cat([head * cos + turned * sin, x[..., n:].float()], dim=-1).type_as(x)


def action_windows(c: Config, actions, start_frame, frames):
    """actions: (batch, video frames, dim), one row per video frame. Latent frame f reads video frames
    4f − 12 … 4f − 1; before the first frame, frame 0's action repeats. → (batch, frames, 12, dim)"""
    span = c.video_frames_per_latent * c.action_window
    padded = torch.cat([actions[:, :1].expand(-1, span, -1), actions], dim=1)  # index = video frame + 12
    step = c.video_frames_per_latent
    return torch.stack([padded[:, step * f : step * f + span] for f in range(start_frame, start_frame + frames)], dim=1)


class MouseAttention(nn.Module):
    """At each of the 880 positions on its own: append the camera moves of the token's 12 video frames to it, then
    attend over time to the same position in the last 6 latent frames, from a cache of its own."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        moves = c.video_frames_per_latent * c.action_window * c.mouse_dim  # 24 numbers per latent frame
        self.mlp = nn.Sequential(
            nn.Linear(c.d_model + moves, c.action_dim), nn.GELU(approximate="tanh"), nn.Linear(c.action_dim, c.action_dim), nn.LayerNorm(c.action_dim)
        )
        self.qkv = nn.Linear(c.action_dim, 3 * c.action_dim, bias=False)
        self.out = nn.Linear(c.action_dim, c.d_model, bias=False)

    def forward(self, x, mouse, start_frame, frames, cache):  # mouse: (batch, video frames so far, 2)
        c, (b, n, d) = self.c, x.shape
        s = n // frames
        tokens = x.view(b, frames, s, d).transpose(1, 2).reshape(b * s, frames, d)  # one row of frames per position
        moves = action_windows(c, mouse, start_frame, frames).flatten(2)  # (batch, frames, 24)
        moves = moves[:, None].expand(b, s, frames, -1).reshape(b * s, frames, -1)
        q, k, v = self.qkv(self.mlp(torch.cat([tokens, moves], dim=-1))).view(b * s, frames, 3, c.action_heads, -1).unbind(2)
        q, k = action_rope(c, rms(q), start_frame), action_rope(c, rms(k), start_frame)
        keys, values = with_cache(cache, k, v, c.window_frames)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
        out = out.transpose(1, 2).reshape(b, s, frames, -1).transpose(1, 2).reshape(b, n, -1)
        return x + self.out(out)


# %% [keyboard] Keyboard attention
class KeyboardAttention(nn.Module):
    """Each token asks which keys were held: its query meets keys and values made from the embedded keyboard
    states of the 12 video frames of each latent frame in the window, the same for all 880 positions."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = nn.Sequential(nn.Linear(c.keyboard_dim, c.keyboard_embed), nn.SiLU(), nn.Linear(c.keyboard_embed, c.keyboard_embed))
        self.q = nn.Linear(c.d_model, c.action_dim, bias=False)
        self.kv = nn.Linear(c.keyboard_embed * c.video_frames_per_latent * c.action_window, 2 * c.action_dim, bias=False)
        self.out = nn.Linear(c.action_dim, c.d_model, bias=False)

    def forward(self, x, keyboard, start_frame, frames, cache):  # keyboard: (batch, video frames so far, 4)
        c, (b, n, d) = self.c, x.shape
        s = n // frames
        held = action_windows(c, self.embed(keyboard), start_frame, frames).flatten(2)  # (batch, frames, 12 × 128)
        k, v = self.kv(held).view(b, frames, 2, c.action_heads, -1).unbind(2)
        q = self.q(x).view(b, frames, s, c.action_heads, -1).transpose(1, 2).reshape(b * s, frames, c.action_heads, -1)
        q, k = action_rope(c, rms(q), start_frame), action_rope(c, rms(k), start_frame)
        keys, values = with_cache(cache, k, v, c.window_frames)
        keys, values = (t[:, None].expand(-1, s, -1, -1, -1).flatten(0, 1) for t in (keys, values))  # every position
        out = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
        out = out.transpose(1, 2).reshape(b, s, frames, -1).transpose(1, 2).reshape(b, n, -1)
        return x + self.out(out)


# %% [mlp] Feed-forward
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [block] Matrix-Game block
class Block(nn.Module):
    """Self-attention and the MLP are modulated per latent frame (shift β, scale γ, gate α from that frame's t);
    in blocks 1–15 the mouse and keyboard attention sit between the image cross-attention and the MLP."""

    def __init__(self, c: Config, index: int):
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 6, c.d_model))  # learned offsets added to the six vectors
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps, elementwise_affine=False)
        self.attention = SelfAttention(c)
        self.cross = ImageCrossAttention(c)
        self.mouse = MouseAttention(c) if index < c.action_blocks else None
        self.keyboard = KeyboardAttention(c) if index < c.action_blocks else None
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps, elementwise_affine=False)
        self.mlp = MLP(c)

    def forward(self, x, modulation, angles, image_tokens, mouse, keyboard, start_frame, cache):
        b, n, d = x.shape
        f = modulation.shape[1]
        shift1, scale1, gate1, shift2, scale2, gate2 = (m[:, :, None] for m in (self.modulation[:, None] + modulation).unbind(2))

        def per_frame(y):
            return y.view(b, f, n // f, d)

        h = (per_frame(self.norm1(x)) * (1 + scale1) + shift1).view(b, n, d)
        x = x + (per_frame(self.attention(h, angles, f, cache["self"])) * gate1).view(b, n, d)
        x = x + self.cross(x, image_tokens)
        if self.mouse is not None:
            x = self.mouse(x, mouse, start_frame, f, cache["mouse"])
            x = self.keyboard(x, keyboard, start_frame, f, cache["keyboard"])
        h = (per_frame(self.norm2(x)) * (1 + scale2) + shift2).view(b, n, d)
        return x + (per_frame(self.mlp(h)) * gate2).view(b, n, d)


# %% [output] Output head and unpatchify
class OutputHead(nn.Module):
    """adaLN with each frame's e (shift and scale, no gate), a Linear to 1 × 2 × 2 × 16 numbers per token, then the
    tokens back into a latent: the predicted velocity v = ε − x₀."""

    def __init__(self, c: Config):
        super().__init__()
        self.channels = c.latent_channels
        self.modulation = nn.Parameter(torch.zeros(1, 2, c.d_model))
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps, elementwise_affine=False)
        self.linear = nn.Linear(c.d_model, 4 * c.latent_channels)

    def forward(self, x, e, grid):  # e: (batch, frames, 1, d_model)
        f, h, w = grid
        shift, scale = (self.modulation[:, None] + e).chunk(2, dim=2)
        x = self.linear(self.norm(x).view(x.shape[0], f, h * w, -1) * (1 + scale) + shift)  # (batch, f, h·w, 64)
        x = x.view(-1, f, h, w, 2, 2, self.channels).permute(0, 6, 1, 2, 4, 3, 5)  # (batch, c, f, h, dy, w, dx)
        return x.reshape(-1, self.channels, f, 2 * h, 2 * w)


# %% [scheduler-step] Few-step flow matching
def flow_sigmas(steps=(1000, 666, 333), shift=5.0, train_steps=1000):
    """The distilled model's three noise levels. Of 1000 training levels σ = 1 … 0.001, warped by the shift
    σ' = 5σ / (1 + 4σ), it uses those at steps 1000, 666 and 333: σ = 1, 0.909, 0.714 (t = 1000 σ)."""
    sigma = torch.linspace(1.0, 0.0, train_steps + 1)[:-1]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma[[train_steps - step for step in steps]]


def flow_step(x_t, velocity, sigma, sigma_next=None, noise=None):
    """x_t = (1 − σ) x₀ + σ ε and the model predicts v = ε − x₀, so x₀ = x_t − σ v. Unless this was the last
    step, x₀ is noised again to the next level with fresh noise (Self-Forcing's few-step sampler)."""
    x0 = x_t - sigma * velocity
    return x0, None if sigma_next is None else (1 - sigma_next) * x0 + sigma_next * noise


# %% [model] Full model and the autoregressive rollout
class MatrixGame(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.patchify = PatchEmbedding(c)
        self.timestep = TimestepEmbedding(c)
        self.image = ImageProjection(c)
        self.blocks = nn.ModuleList(Block(c, i) for i in range(c.n_layers))
        self.head = OutputHead(c)

    def forward(self, noisy, t, condition, clip_tokens, mouse, keyboard, start_frame, caches):
        x, grid = self.patchify(noisy, condition)
        e, modulation = self.timestep(t)
        angles = rope_angles(self.c, *grid, start_frame)
        image_tokens = self.image(clip_tokens)
        for block, cache in zip(self.blocks, caches):
            x = block(x, modulation, angles, image_tokens, mouse, keyboard, start_frame, cache)
        return self.head(x, e, grid)  # velocity, (batch, 16, frames, 44, 80)


def commit(caches, window):
    """After the clean pass, the chunk's keys and values join every cache; only the last \`window\` frames stay."""
    for block in caches:
        for cache in block.values():
            if "new" not in cache:  # blocks 16–30 have no action module
                continue
            k, v = cache.pop("new")
            cache["k"] = torch.cat([cache["k"], k], dim=1)[:, -window:] if "k" in cache else k
            cache["v"] = torch.cat([cache["v"], v], dim=1)[:, -window:] if "v" in cache else v


@torch.no_grad()
def rollout(model, image_latent, clip_tokens, mouse, keyboard, noise, fresh=torch.randn):
    """Generate chunk after chunk. Each chunk: 3 latent frames from noise in 3 flow steps, then one clean pass at
    t = 0 that writes the chunk into every cache. mouse (…, 2) and keyboard (…, 4) have one row per video frame;
    fresh(shape) draws the noise that re-noises x₀ between steps."""
    c, (b, channels, frames, h, w) = model.c, noise.shape
    condition = first_frame_condition(image_latent)
    sigmas = flow_sigmas()
    caches = [{"self": {}, "mouse": {}, "keyboard": {}} for _ in model.blocks]
    chunks = []
    for start in range(0, frames, c.frames_per_chunk):
        n = c.frames_per_chunk
        x, cond = noise[:, :, start : start + n], condition[:, :, start : start + n]
        seen = 1 + c.video_frames_per_latent * (start + n - 1)  # video frames up to the chunk's last
        actions = (mouse[:, :seen], keyboard[:, :seen])
        for i, sigma in enumerate(sigmas):
            velocity = model(x, torch.full((b, n), sigma.item() * 1000), cond, clip_tokens, *actions, start, caches)
            last = i + 1 == len(sigmas)
            x0, x = flow_step(x, velocity, sigma, *(() if last else (sigmas[i + 1], fresh((b, channels, n, h, w)))))
        model(x0, torch.zeros(b, n), cond, clip_tokens, *actions, start, caches)  # the clean pass
        commit(caches, c.window_frames)
        chunks.append(x0)
    return torch.cat(chunks, dim=2)  # x₀ of every latent frame; the VAE decodes it to 4 video frames each


# Skywork's checkpoint: model = MatrixGame(); model.load_state_dict(...) after renaming its keys (checks/diffusion_matrix_game.py).
# video = rollout(model, vae_latent_of_image_and_gray_frames, clip_tokens, mouse, keyboard, torch.randn(1, 16, 12, 44, 80))
`;export{e as default};
