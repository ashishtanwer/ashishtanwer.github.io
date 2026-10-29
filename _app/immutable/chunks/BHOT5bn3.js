const n=`# Octo-Small 1.5 (rail-berkeley/octo-small-1.5), the model in the Embodied Explainer's Octo tab, written as plain
# PyTorch blocks. The released model is JAX/Flax; each "# %% [id]" section is checked against Octo's own Flax code.
# Tokenizers turn the camera image (and a goal image, black when the task is language), and the instruction (through a
# frozen T5-Base encoder) into 384-wide tokens. A transformer with a block-wise mask reads [instruction, then per
# timestep: image tokens, the instruction again, one readout token]; nothing attends to a readout. The readout's output
# conditions a small diffusion network that denoises a chunk of 4 future actions in 20 steps.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # Image tokenizer: SmallStem16
    image_size: int = 256  # primary camera; the wrist camera (128) has its own stem, unused on Bridge
    stem_channels: tuple = (32, 96, 192, 384)  # four 3 × 3 stride-2 convolutions: 256 → 16
    stem_features: int = 512  # a 1 × 1 convolution per 16 × 16 cell: 256 tokens of 512
    goal_channels: int = 3  # the goal image is stacked as 3 more input channels
    # Language tokenizer: T5-Base encoder, frozen
    vocab_size: int = 32128
    t5_d_model: int = 768
    t5_heads: int = 12
    t5_layers: int = 12
    t5_d_ff: int = 3072
    t5_buckets: int = 32
    t5_max_distance: int = 128
    text_tokens: int = 16  # instructions are padded or cut to 16 tokens
    # Transformer
    d_model: int = 384
    n_heads: int = 6  # 64 dimensions per head
    n_layers: int = 12
    d_mlp: int = 1536
    max_horizon: int = 10  # learned positions for up to 10 timesteps; the window is 2
    eps: float = 1e-6
    # Diffusion action head
    action_dim: int = 7
    action_horizon: int = 4  # a chunk of 4 actions
    time_dim: int = 32
    hidden_dim: int = 256
    blocks: int = 3
    diffusion_steps: int = 20
    max_action: float = 5.0


# %% [image-tokenizer] Image tokenizer
class StdConv(nn.Conv2d):
    """A convolution whose kernel is standardized (zero mean, unit std per output channel) on every call."""

    def forward(self, x):
        w = self.weight
        mean = w.mean(dim=(1, 2, 3), keepdim=True)
        std = w.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return F.conv2d(x, (w - mean) / (std + 1e-5), self.bias, self.stride, self.padding)


class ImageTokenizer(nn.Module):
    """Pixels to [-1, 1]; observation and goal stacked as channels; four StdConv → GroupNorm → ReLU stages halve
    the size; a 1 × 1 convolution makes one 512-wide token per 16 × 16 cell; a projection to 384 and a learned
    position per timestep and cell."""

    def __init__(self, c: Config, image_size: int = 256):
        super().__init__()
        channels = [3 + c.goal_channels, *c.stem_channels]
        self.stem = nn.ModuleList(
            nn.ModuleDict({"conv": StdConv(cin, cout, 3, stride=2, padding=1), "norm": nn.GroupNorm(32, cout, c.eps)})
            for cin, cout in zip(channels[:-1], channels[1:])
        )
        self.embedding = nn.Conv2d(c.stem_channels[-1], c.stem_features, kernel_size=1)
        self.projection = nn.Linear(c.stem_features, c.d_model)
        side = image_size // 16
        self.position = nn.Parameter(torch.zeros(1, c.max_horizon, side * side, c.d_model))

    def forward(self, images, goal):  # images: (batch, window, 3, H, W) uint8; goal: (batch, 3, H, W)
        b, t = images.shape[:2]
        x = torch.cat([images, goal[:, None].expand(-1, t, -1, -1, -1)], dim=2).flatten(0, 1)
        x = x.float() / 127.5 - 1.0  # (batch · window, 6, 256, 256)
        for stage in self.stem:
            x = F.relu(stage["norm"](stage["conv"](x)))  # 128 → 64 → 32 → 16
        x = self.embedding(x).flatten(2).transpose(1, 2)  # (batch · window, 256, 512), cells in reading order
        x = self.projection(x).view(b, t, -1, self.projection.out_features)
        return x + self.position[:, :t]  # (batch, window, 256, 384)


# %% [t5] T5-Base encoder
class T5Attention(nn.Module):
    """T5 self-attention: no 1/√d scaling, and a learned bias per head for each bucket of relative distance
    (only the first layer has the table; every layer adds the same bias)."""

    def __init__(self, c: Config, has_bias: bool):
        super().__init__()
        self.c = c
        inner = c.t5_d_model
        self.q, self.k, self.v, self.o = (nn.Linear(c.t5_d_model, inner, bias=False) for _ in range(4))
        self.bias = nn.Embedding(c.t5_buckets, c.t5_heads) if has_bias else None

    def position_bias(self, n):
        c = self.c
        relative = torch.arange(n)[None, :] - torch.arange(n)[:, None]  # key − query
        half = c.t5_buckets // 2  # bidirectional: half the buckets for each direction
        bucket = (relative > 0).long() * half
        distance = relative.abs()
        exact = half // 2
        log = exact + (
            torch.log(distance.float().clamp(min=1) / exact) / math.log(c.t5_max_distance / exact) * (half - exact)
        ).long()
        bucket += torch.where(distance < exact, distance, log.clamp(max=half - 1))
        return self.bias(bucket).permute(2, 0, 1)[None]  # (1, heads, n, n)

    def forward(self, x, mask_bias, bias):
        b, n, _ = x.shape
        q, k, v = (proj(x).view(b, n, self.c.t5_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = (q @ k.transpose(-2, -1) + bias + mask_bias).softmax(dim=-1)
        return self.o((weights @ v).transpose(1, 2).reshape(b, n, -1))


class T5Norm(nn.Module):
    """T5's layer norm: scale only, no mean subtraction (an RMSNorm)."""

    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class T5Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.t5_d_model)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm1": T5Norm(c.t5_d_model, c.eps),
                    "attention": T5Attention(c, has_bias=i == 0),
                    "norm2": T5Norm(c.t5_d_model, c.eps),
                    "wi": nn.Linear(c.t5_d_model, c.t5_d_ff, bias=False),
                    "wo": nn.Linear(c.t5_d_ff, c.t5_d_model, bias=False),
                }
            )
            for i in range(c.t5_layers)
        )
        self.norm = T5Norm(c.t5_d_model, c.eps)

    def forward(self, ids, mask):  # ids, mask: (batch, 16); mask is 0 on padding
        x = self.embed(ids)
        mask_bias = (1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        bias = self.layers[0]["attention"].position_bias(ids.shape[1])
        for layer in self.layers:
            x = x + layer["attention"](layer["norm1"](x), mask_bias, bias)
            x = x + layer["wo"](F.relu(layer["wi"](layer["norm2"](x))))
        return self.norm(x)  # (batch, 16, 768)


# %% [language-tokenizer] Language tokenizer
class LanguageTokenizer(nn.Module):
    """The frozen T5 encoder's 16 outputs, projected to 384, plus a learned position each."""

    def __init__(self, c: Config):
        super().__init__()
        self.t5 = T5Encoder(c)
        self.projection = nn.Linear(c.t5_d_model, c.d_model)
        self.position = nn.Parameter(torch.zeros(1, c.text_tokens, c.d_model))

    def forward(self, ids, mask):
        return self.projection(self.t5(ids, mask)) + self.position  # (batch, 16, 384)


# %% [attention-mask] Block-wise attention mask
def block_mask(has_text, window: int, image_tokens: int, text_tokens: int = 16):
    """The sequence is [instruction (16)] then, per timestep, [image tokens, the instruction again, 1 readout].
    A token sees the instruction, and every image or instruction token of its own or earlier timesteps. A readout
    also sees itself (and earlier readouts); no other token sees a readout. \`has_text\` (batch,) hides the
    instruction's tokens when a task has none; padding inside the instruction is not hidden (Octo's language mask
    comes from whether the instruction exists, so all 16 T5 outputs are read). Returns (batch, 1, n, n) booleans,
    True where attention is allowed."""
    b = has_text.shape[0]
    # Group of every token: 0 instruction prefix, 1 image, 2 repeated instruction, 3 readout; and its timestep.
    groups = [0] * text_tokens + ([1] * image_tokens + [2] * text_tokens + [3]) * window
    steps = [-1] * text_tokens + sum(([t] * (image_tokens + text_tokens + 1) for t in range(window)), [])
    group, step = torch.tensor(groups), torch.tensor(steps)
    earlier = step[None, :] <= step[:, None]  # key's timestep ≤ query's
    to_input = (group[None, :] != 3) & earlier  # instruction and image tokens, causally in time
    to_readout = (group[:, None] == 3) & (group[None, :] == 3) & earlier
    allowed = to_input | to_readout
    allowed &= (group[:, None] != 0) | (group[None, :] == 0)  # the prefix sees only the prefix
    is_text = (group == 0) | (group == 2)
    keep = ~is_text[None, :] | has_text.bool()[:, None]  # (batch, n): keys that exist
    return allowed[None, None] & keep[:, None, None, :]


# %% [attention] Masked self-attention
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q, self.k, self.v, self.out = (nn.Linear(c.d_model, c.d_model) for _ in range(4))

    def forward(self, x, mask):  # x: (batch, n, 384); mask: (batch, 1, n, n)
        b, n, _ = x.shape
        q, k, v = (proj(x).view(b, n, self.c.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, -1))


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))  # Flax's nn.gelu is the tanh form


# %% [block] Transformer block
class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


# %% [transformer] Octo transformer
class OctoTransformer(nn.Module):
    """Assembles the sequence, runs 12 blocks under the block-wise mask, and returns the readout tokens."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.image = ImageTokenizer(c)
        self.language = LanguageTokenizer(c)
        self.readout = nn.Parameter(torch.zeros(1, c.max_horizon, 1, c.d_model))  # a readout is only its position
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, images, goal, ids, text_mask):
        b, t = images.shape[:2]
        text = self.language(ids, text_mask)  # (batch, 16, 384)
        image = self.image(images, goal)  # (batch, window, 256, 384)
        per_step = torch.cat([image, text[:, None].expand(-1, t, -1, -1), self.readout[:, :t].expand(b, -1, -1, -1)], 2)
        x = torch.cat([text, per_step.flatten(1, 2)], dim=1)  # (batch, 16 + window · 273, 384)
        mask = block_mask(torch.ones(b, dtype=torch.bool), t, image.shape[2], text.shape[1])
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        readouts = x[:, text.shape[1] :].view(b, t, -1, x.shape[-1])[:, :, -1]
        return readouts  # (batch, window, 384): one readout per timestep


# %% [diffusion-head] Diffusion action head
class DiffusionHead(nn.Module):
    """ε̂(readout, x_t, t): learned Fourier features of t → MLP → concatenated with the readout and the noisy
    chunk → a residual MLP with LayerNorm and swish → 28 numbers (4 actions × 7)."""

    def __init__(self, c: Config):
        super().__init__()
        out = c.action_dim * c.action_horizon
        self.fourier = nn.Parameter(torch.zeros(c.time_dim // 2, 1))
        self.time1 = nn.Linear(c.time_dim, 2 * c.time_dim)
        self.time2 = nn.Linear(2 * c.time_dim, c.time_dim)
        self.inp = nn.Linear(c.time_dim + c.d_model + out, c.hidden_dim)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(c.hidden_dim, eps=c.eps),
                    "up": nn.Linear(c.hidden_dim, 4 * c.hidden_dim),
                    "down": nn.Linear(4 * c.hidden_dim, c.hidden_dim),
                }
            )
            for _ in range(c.blocks)
        )
        self.out = nn.Linear(c.hidden_dim, out)

    def forward(self, readout, x, time):  # readout: (..., 384); x: (..., 28); time: (..., 1)
        f = 2 * math.pi * time @ self.fourier.T
        cond = self.time2(F.silu(self.time1(torch.cat([f.cos(), f.sin()], dim=-1))))
        h = self.inp(torch.cat([cond, readout, x], dim=-1))
        for block in self.blocks:
            h = h + block["down"](F.silu(block["up"](block["norm"](h))))
        return self.out(F.silu(h))


# %% [ddpm] Denoising the action chunk
def cosine_betas(steps: int, s: float = 0.008):
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return (1 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(0, 0.999).float()


@torch.no_grad()
def sample(head: DiffusionHead, readout, noise, zs, c: Config = Config()):
    """DDPM from x_T = noise: x ← (x − β/√(1−ᾱ) · ε̂) / √α, plus √β · z except at the last step, clipped to ±5.
    \`zs\` are the fresh noises of the 20 steps (in Octo they come from the JAX random key)."""
    betas = cosine_betas(c.diffusion_steps)
    alphas = 1 - betas
    alpha_hats = alphas.cumprod(0)
    x = noise
    for i, time in enumerate(range(c.diffusion_steps - 1, -1, -1)):
        eps = head(readout, x, torch.full((*x.shape[:-1], 1), float(time)))
        x = (x - (1 - alphas[time]) / (1 - alpha_hats[time]).sqrt() * eps) / alphas[time].sqrt()
        if time > 0:
            x = x + betas[time].sqrt() * zs[i]
        x = x.clamp(-c.max_action, c.max_action)
    return x.view(*x.shape[:-1], c.action_horizon, c.action_dim)  # (..., 4, 7)


# %% [unnormalize] Back to the robot's units
def unnormalize(actions, mean, std, mask=None):
    """Octo's actions are normalized per dataset to zero mean and unit std (Bridge: bridge_dataset's)."""
    out = actions * std + mean
    return out if mask is None else torch.where(mask, out, actions)


# %% [model] Full model
class Octo(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.transformer = OctoTransformer(c)
        self.head = DiffusionHead(c)

    @torch.no_grad()
    def sample_actions(self, images, ids, text_mask, noise, zs, mean, std, mask=None):
        goal = torch.zeros_like(images[:, 0])  # language task: the goal image is all zeros (black)
        readout = self.transformer(images, goal, ids, text_mask)[:, -1]  # the newest timestep's readout
        return unnormalize(sample(self.head, readout, noise, zs, self.c), mean, std, mask)  # (batch, 4, 7)


# images: (1, 2, 3, 256, 256) uint8, the two newest camera frames; ids, text_mask: T5 tokens of the instruction
`;export{n as default};
