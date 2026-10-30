const e=`# OpenVLA (openvla/openvla-7b-finetuned-libero-spatial), the model in the Embodied Explainer's OpenVLA tab,
# written as plain PyTorch blocks. Each "# %% [id]" section is one block of the diagram.
# Two vision encoders (DINOv2 ViT-L/14 with registers, SigLIP So400m/14) read the same 224 × 224 image; their
# patch features are concatenated, projected into Llama 2's width and placed after <s>. Llama 2 7B then writes
# seven tokens, one per action dimension, from the 256 least-used tokens of its vocabulary; each token is a bin
# of [-1, 1] that the dataset's statistics turn back into a robot action.
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 14  # 16 × 16 = 256 patches
    d_model: int = 1024
    heads: int = 16
    layers: int = 24
    d_mlp: int = 4096
    class_token: bool = True  # DINOv2 has a [CLS] token and 4 registers in front of the patches;
    registers: int = 4  # SigLIP has neither
    layer_scale: bool = True  # DINOv2 scales each residual branch by a learned vector; SigLIP does not
    eps: float = 1e-6


@dataclass
class Config:
    dino: VisionConfig = field(default_factory=VisionConfig)
    siglip: VisionConfig = field(
        default_factory=lambda: VisionConfig(
            d_model=1152, layers=27, d_mlp=4304, class_token=False, registers=0, layer_scale=False
        )
    )
    # Llama 2 7B
    vocab_size: int = 32064  # 32,000 tokens + <PAD>, padded to a multiple of 64
    d_model: int = 4096
    n_heads: int = 32  # 128 dimensions per head
    n_layers: int = 32
    d_mlp: int = 11008
    rope_theta: float = 10_000.0
    eps: float = 1e-6
    # Actions
    action_dim: int = 7  # Δx Δy Δz Δroll Δpitch Δyaw gripper
    n_bins: int = 256
    text_vocab: int = 32000  # action tokens are ids 31744 … 31999, the last 256 of Llama's vocabulary
    empty_token: int = 29871  # '▁', appended after "Out:"


# %% [patch-embedding] Patch embedding
class PatchEmbedding(nn.Module):
    """A 14 × 14 convolution with stride 14 turns each patch into one vector; a learned position is added to
    each patch, then DINOv2's [CLS] and 4 register tokens go in front (they get no position)."""

    def __init__(self, c: VisionConfig):
        super().__init__()
        n = (c.image_size // c.patch_size) ** 2
        self.patch = nn.Conv2d(3, c.d_model, kernel_size=c.patch_size, stride=c.patch_size)
        self.position = nn.Parameter(torch.zeros(1, n, c.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, c.d_model)) if c.class_token else None
        self.registers = nn.Parameter(torch.zeros(1, c.registers, c.d_model)) if c.registers else None

    def forward(self, pixels):  # (images, 3, 224, 224)
        x = self.patch(pixels).flatten(2).transpose(1, 2) + self.position  # (images, 256, d)
        prefix = [p.expand(x.shape[0], -1, -1) for p in (self.cls, self.registers) if p is not None]
        return torch.cat(prefix + [x], dim=1)  # (images, 1 + 4 + 256, d) for DINOv2


# %% [vision-block] Vision block
class VisionBlock(nn.Module):
    """Pre-LayerNorm block: every token attends to every token, then a GELU MLP."""

    def __init__(self, c: VisionConfig):
        super().__init__()
        self.heads = c.heads
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)
        self.scale1 = nn.Parameter(torch.ones(c.d_model)) if c.layer_scale else None
        self.scale2 = nn.Parameter(torch.ones(c.d_model)) if c.layer_scale else None

    def attention(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)  # each (b, heads, t, 64 or 72)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, d))

    def forward(self, x):
        a = self.attention(self.norm1(x))
        x = x + (a * self.scale1 if self.scale1 is not None else a)
        m = self.down(F.gelu(self.up(self.norm2(x))))
        return x + (m * self.scale2 if self.scale2 is not None else m)


# %% [vision-encoder] Patch features
class VisionEncoder(nn.Module):
    """OpenVLA takes the second-to-last block's output, without the final LayerNorm, and drops the prefix
    tokens: 256 patch vectors. (The last block still runs; its output is not used.)"""

    def __init__(self, c: VisionConfig):
        super().__init__()
        self.prefix_tokens = int(c.class_token) + c.registers
        self.embedding = PatchEmbedding(c)
        self.blocks = nn.ModuleList(VisionBlock(c) for _ in range(c.layers))

    def forward(self, pixels):
        x = self.embedding(pixels)
        for block in self.blocks[:-1]:
            x = block(x)
        return x[:, self.prefix_tokens :]  # (images, 256, d)


# %% [fusion] Two encoders, one feature per patch
class FusedVision(nn.Module):
    """The processor normalizes the image twice (ImageNet's mean and std for DINOv2, 0.5 for SigLIP) and stacks
    the copies as 6 channels; each encoder reads its own copy, and their features are concatenated per patch."""

    def __init__(self, c: Config):
        super().__init__()
        self.dino = VisionEncoder(c.dino)
        self.siglip = VisionEncoder(c.siglip)

    def forward(self, pixels):  # (images, 6, 224, 224)
        dino_pixels, siglip_pixels = pixels.split([3, 3], dim=1)
        return torch.cat([self.dino(dino_pixels), self.siglip(siglip_pixels)], dim=2)  # (images, 256, 2176)


# %% [projector] Projector
class Projector(nn.Module):
    """A three-layer GELU MLP from 2,176 to Llama's 4,096: each patch becomes one image token."""

    def __init__(self, c: Config):
        super().__init__()
        d_vision = c.dino.d_model + c.siglip.d_model
        self.fc1 = nn.Linear(d_vision, 4 * d_vision)  # 2176 → 8704
        self.fc2 = nn.Linear(4 * d_vision, c.d_model)  # 8704 → 4096
        self.fc3 = nn.Linear(c.d_model, c.d_model)

    def forward(self, features):
        return self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(features)))))  # (images, 256, 4096)


# %% [embedding] Image tokens after <s>
class Embedding(nn.Module):
    """Text tokens look up a row; the 256 image tokens are inserted right after <s>, before the prompt."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)

    def forward(self, ids, image_tokens):  # ids: (batch, seq) starting with <s>; image_tokens: (batch, 256, 4096)
        x = self.token(ids)
        return torch.cat([x[:, :1], image_tokens, x[:, 1:]], dim=1)  # (batch, 1 + 256 + seq - 1, 4096)


# %% [rope] RoPE
def rope(x, theta, start=0):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    seq, d = x.shape[-2], x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(start, start + seq, device=x.device)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Causal self-attention
class Attention(nn.Module):
    """32 heads, each with its own query, key and value; RoPE on queries and keys; each position sees itself
    and everything before it, image tokens included."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q, self.k, self.v, self.out = (nn.Linear(c.d_model, c.d_model, bias=False) for _ in range(4))

    def forward(self, x):  # x: (batch, seq, 4096)
        c, (b, t, _) = self.c, x.shape
        q, k, v = (proj(x).view(b, t, c.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (batch, heads, seq, seq)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [mlp] SwiGLU MLP
class MLP(nn.Module):
    """SiLU(gate) × up, then down: 4096 → 11008 → 4096."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-RMSNorm residual block, as in every Llama 2 layer."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [output] Next-token logits
class Output(nn.Module):
    """Final RMSNorm, then one logit per vocabulary token. The next token is the argmax over all 32,064, not
    only the action tokens; the fine-tuned model puts its top logit on an action token at every step."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # not tied to the embedding

    def forward(self, x):
        return self.lm_head(self.norm(x))  # (batch, seq, 32064)


# %% [action-tokens] Action bins and tokens
def bin_edges(c: Config = Config()):
    """256 edges from -1 to 1, computed as numpy.linspace does (start + i · step, last = stop), so a value on
    an edge falls in the same bin as in OpenVLA; torch.linspace can differ in the last bit."""
    edges = -1 + torch.arange(c.n_bins, dtype=torch.float64) * (2 / (c.n_bins - 1))
    edges[-1] = 1
    return edges


def action_to_tokens(normalized, c: Config = Config()):
    """Training: each dimension in [-1, 1] falls into one of 256 bins and becomes token text_vocab - bin."""
    bins = torch.bucketize(normalized.clamp(-1, 1).double(), bin_edges(c), right=True)  # 1 … 256, as np.digitize
    return c.text_vocab - bins


def tokens_to_action(ids, c: Config = Config()):
    """Inference: token id → bin index (0 … 254) → that bin's center in [-1, 1]."""
    edges = bin_edges(c)
    centers = (edges[:-1] + edges[1:]) / 2  # 255 centers between 256 edges
    bins = (c.text_vocab - ids - 1).clamp(0, centers.numel() - 1)
    return centers[bins]


# %% [unnormalize] Back to the robot's units
def unnormalize(normalized, q01, q99, mask=None):
    """[-1, 1] → [q01, q99] of the fine-tuning dataset's actions, per dimension. Masked-out dimensions (the
    gripper in some datasets) stay as they are."""
    actions = 0.5 * (normalized + 1) * (q99 - q01) + q01
    return actions if mask is None else torch.where(mask, actions, normalized)


# %% [model] Full model
class OpenVLA(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.vision = FusedVision(c)
        self.projector = Projector(c)
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)

    def logits(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x)

    @torch.no_grad()
    def predict_action(self, pixels, ids, q01, q99, mask=None):
        """Greedy decoding of action_dim tokens (no KV cache: the whole sequence is run again for each token),
        then tokens → bin centers → actions."""
        if ids[0, -1] != self.c.empty_token:
            ids = torch.cat([ids, ids.new_full((ids.shape[0], 1), self.c.empty_token)], dim=1)
        x = self.embedding(ids, self.projector(self.vision(pixels)))
        tokens = []
        for _ in range(self.c.action_dim):
            token = self.logits(x)[:, -1].argmax(dim=-1)  # (batch,)
            tokens.append(token)
            x = torch.cat([x, self.embedding.token(token)[:, None]], dim=1)
        tokens = torch.stack(tokens, dim=1)  # (batch, 7)
        return tokens, unnormalize(tokens_to_action(tokens, self.c), q01, q99, mask)


# tokens, actions = OpenVLA().predict_action(pixels, ids, q01, q99)  # pixels: (1, 6, 224, 224)
# ids: "<s> In: What action should the robot take to pick up the black bowl …?\\nOut:"
`;export{e as default};
