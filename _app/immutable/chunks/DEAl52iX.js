const e=`# SmolVLM-256M-Instruct (HuggingFaceTB/SmolVLM-256M-Instruct), the model in the Vision–language tab,
# written as plain PyTorch blocks. Each "# %% [id]" section is one block of the diagram.
# A SigLIP-style vision encoder reads the image, a connector packs its patches into 16 image tokens,
# and a Llama-style decoder reads those tokens in place of the <image> slots in the prompt.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # Vision encoder
    image_size: int = 512  # the position table covers 512 × 512; the tab's image is 256 × 256
    patch_size: int = 16
    channels: int = 3
    vision_d_model: int = 768
    vision_heads: int = 12  # 64 dimensions per head
    vision_layers: int = 12
    vision_d_mlp: int = 3072
    vision_eps: float = 1e-6
    # Connector
    shuffle: int = 4  # each image token packs a 4 × 4 block of patches: 16 × 768 = 12,288 numbers
    # Language decoder
    vocab_size: int = 49280
    d_model: int = 576
    n_heads: int = 9  # 64 dimensions per head
    n_kv_heads: int = 3  # grouped-query attention: three query heads share each key/value head
    n_layers: int = 30
    d_mlp: int = 1536
    rope_theta: float = 100_000.0
    eps: float = 1e-5
    image_token: int = 49190  # the <image> id the connector output replaces


# %% [patch-embedding] Vision embedding
class PatchEmbedding(nn.Module):
    """Wpatch on every 16 × 16 patch, plus a learned position looked up for each patch."""

    def __init__(self, c: Config):
        super().__init__()
        self.side = c.image_size // c.patch_size  # 32: the position table is 32 × 32 = 1,024 rows
        # A 16 × 16 convolution with stride 16 flattens each patch to 768 numbers and multiplies by Wpatch.
        self.patch = nn.Conv2d(c.channels, c.vision_d_model, kernel_size=c.patch_size, stride=c.patch_size)
        self.position = nn.Embedding(self.side**2, c.vision_d_model)

    def position_ids(self, rows, cols):
        """A smaller grid is spread over the 32 × 32 table: each patch takes the row and column bucket its
        fractional coordinate falls in. A 16 × 16 grid gets rows and columns 0, 1, 3, 5, …, 29."""
        boundaries = torch.arange(1 / self.side, 1.0, 1 / self.side)

        def buckets(n):
            return torch.bucketize(torch.arange(n) / n * (1 - 1e-6), boundaries, right=True)

        return (buckets(rows)[:, None] * self.side + buckets(cols)).flatten()

    def forward(self, pixels):  # pixels: (images, 3, 256, 256)
        grid = self.patch(pixels)  # (images, 768, 16, 16)
        ids = self.position_ids(grid.shape[2], grid.shape[3]).to(pixels.device)
        return grid.flatten(2).transpose(1, 2) + self.position(ids)  # (images, 256, 768)


# %% [rope] RoPE
def rope(x, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    seq, d = x.shape[-2], x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(seq, device=x.device)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Self-attention
class VisionAttention(nn.Module):
    """Vision encoder: every patch attends to every patch; no mask and no rotation."""

    def __init__(self, c: Config):
        super().__init__()
        d = c.vision_d_model
        self.n_heads = c.vision_heads
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):  # x: (images, patches, 768)
        b, t, d = x.shape
        q, k, v = (proj(x).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, d))


class Attention(nn.Module):
    """Language decoder: causal grouped-query attention, with RoPE on queries and keys."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        head_dim = c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * head_dim, c.d_model, bias=False)

    def forward(self, x):  # x: (batch, seq, 576)
        c, (b, t, _) = self.c, x.shape
        q = self.q(x).view(b, t, c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = self.k(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        q, k = rope(q, c.rope_theta), rope(k, c.rope_theta)
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [vision-mlp] Vision MLP
class VisionMLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.vision_d_model, c.vision_d_mlp)
        self.down = nn.Linear(c.vision_d_mlp, c.vision_d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


# %% [mlp] SwiGLU MLP
class MLP(nn.Module):
    """Language decoder: SiLU(gate) × up, then down."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [block] Transformer block
class VisionBlock(nn.Module):
    """Vision encoder: pre-LayerNorm residual block."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.vision_d_model, eps=c.vision_eps)
        self.attention = VisionAttention(c)
        self.ln_2 = nn.LayerNorm(c.vision_d_model, eps=c.vision_eps)
        self.mlp = VisionMLP(c)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Block(nn.Module):
    """Language decoder: pre-RMSNorm residual block."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [vision-encoder] Vision encoder
class VisionEncoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embedding = PatchEmbedding(c)
        self.blocks = nn.ModuleList(VisionBlock(c) for _ in range(c.vision_layers))
        self.ln_post = nn.LayerNorm(c.vision_d_model, eps=c.vision_eps)

    def forward(self, pixels):
        x = self.embedding(pixels)
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)  # (images, 256, 768)


# %% [pixel-shuffle] Pixel shuffle
def pixel_shuffle(x, s):
    """Pack each s × s block of neighbouring patches into one long vector: 16 × 16 → 4 × 4 tokens."""
    b, n, d = x.shape  # (images, 256, 768)
    side = math.isqrt(n)
    grid = x.view(b, side // s, s, side // s, s, d)  # (images, block row, row in block, block col, col in block, d)
    blocks = grid.permute(0, 1, 3, 2, 4, 5)  # (images, block row, block col, row in block, col in block, d)
    return blocks.reshape(b, (side // s) ** 2, s * s * d)  # (images, 16, 12288), blocks in reading order


# %% [connector] Vision → language
class Connector(nn.Module):
    """Pixel shuffle, then one projection Wproj from 12,288 numbers to the decoder's width."""

    def __init__(self, c: Config):
        super().__init__()
        self.shuffle = c.shuffle
        self.projection = nn.Linear(c.shuffle**2 * c.vision_d_model, c.d_model, bias=False)

    def forward(self, patches):  # (images, 256, 768)
        return self.projection(pixel_shuffle(patches, self.shuffle))  # image tokens: (images, 16, 576)


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Text tokens look up a row; each <image> slot takes the next image token from the connector."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.image_token = c.image_token

    def forward(self, ids, image_tokens):  # ids: (batch, seq); image_tokens: (images, 16, 576)
        x = self.token(ids)
        x[ids == self.image_token] = image_tokens.reshape(-1, x.shape[-1])  # in order, unchanged
        return x  # (batch, seq, 576); positions enter later, through RoPE


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm, then a projection back to one logit per vocabulary token."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding

    def forward(self, x):
        return self.lm_head(self.norm(x))  # logits: (batch, seq, vocab)


def next_token_probabilities(logits, temperature=1.0, top_k=None, top_p=None):
    """The last position's logits, scaled by temperature and cut to top-k or top-p, as probabilities."""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        p = sorted_logits.softmax(dim=-1)
        drop = p.cumsum(dim=-1) - p >= top_p  # keep the smallest set whose mass reaches top_p
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits.softmax(dim=-1)


# %% [model] Full model
class SmolVLM(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.vision_encoder = VisionEncoder(c)
        self.connector = Connector(c)
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, pixels, ids):  # ids hold 16 <image> slots per image
        x = self.embedding(ids, self.connector(self.vision_encoder(pixels)))
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# probabilities = next_token_probabilities(SmolVLM()(pixels, ids), temperature=0.8, top_k=5)
`;export{e as default};
