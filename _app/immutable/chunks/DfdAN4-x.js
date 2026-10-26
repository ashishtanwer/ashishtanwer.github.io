const e=`# CLIP ViT-B/32 (openai/clip-vit-base-patch32), the model in the CLIP tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram. Both encoders use the same block, at different sizes.
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class Encoder:
    d_model: int
    n_heads: int  # 64 dimensions per head in both encoders
    n_layers: int
    d_mlp: int  # 4 × d_model
    eps: float = 1e-5


@dataclass
class Config:
    image_size: int = 224
    patch_size: int = 32  # 224 / 32 = a 7 × 7 grid of 49 patches
    channels: int = 3
    vision: Encoder = field(default_factory=lambda: Encoder(d_model=768, n_heads=12, n_layers=12, d_mlp=3072))
    vocab_size: int = 49408
    max_positions: int = 77
    text: Encoder = field(default_factory=lambda: Encoder(d_model=512, n_heads=8, n_layers=12, d_mlp=2048))
    d_shared: int = 512  # both encoders project into this space
    logit_scale: float = math.log(1 / 0.07)  # initial value; trained CLIP stores log 100 = 4.6052


# %% [attention] Multi-head self-attention
class Attention(nn.Module):
    """Bidirectional in the image encoder; causal in the text encoder, so each token sees only its left."""

    def __init__(self, e: Encoder, causal: bool):
        super().__init__()
        self.n_heads, self.causal = e.n_heads, causal
        self.q = nn.Linear(e.d_model, e.d_model)
        self.k = nn.Linear(e.d_model, e.d_model)
        self.v = nn.Linear(e.d_model, e.d_model)
        self.out = nn.Linear(e.d_model, e.d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        # Split into heads: (batch, heads, seq, head_dim)
        q, k, v = (proj(x).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if self.causal:
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        heads = scores.softmax(dim=-1) @ v  # (batch, heads, seq, head_dim)
        return self.out(heads.transpose(1, 2).reshape(b, t, d))


# %% [mlp] MLP
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)  # CLIP's sigmoid approximation of GELU


class MLP(nn.Module):
    def __init__(self, e: Encoder):
        super().__init__()
        self.up = nn.Linear(e.d_model, e.d_mlp)
        self.down = nn.Linear(e.d_mlp, e.d_model)

    def forward(self, x):
        return self.down(quick_gelu(self.up(x)))


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, e: Encoder, causal: bool):
        super().__init__()
        self.ln_1 = nn.LayerNorm(e.d_model, eps=e.eps)
        self.attention = Attention(e, causal)
        self.ln_2 = nn.LayerNorm(e.d_model, eps=e.eps)
        self.mlp = MLP(e)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


# %% [patch-embedding] Image embedding
class ImageEmbedding(nn.Module):
    """Wpatch on each 32 × 32 patch, a learned [CLS] in front, + position, then a LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        d = c.vision.d_model
        # A 32 × 32 convolution with stride 32 flattens each patch to 32 · 32 · 3 = 3072 numbers
        # and multiplies by Wpatch (3072 → 768). CLIP's patch projection has no bias.
        self.patch = nn.Conv2d(c.channels, d, kernel_size=c.patch_size, stride=c.patch_size, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(d))
        self.position = nn.Embedding((c.image_size // c.patch_size) ** 2 + 1, d)
        self.ln = nn.LayerNorm(d, eps=c.vision.eps)  # CLIP's extra LayerNorm before the first block

    def forward(self, pixels):  # pixels: (batch, 3, 224, 224)
        patches = self.patch(pixels).flatten(2).transpose(1, 2)  # (batch, 49, 768)
        cls = self.cls_token.expand(patches.shape[0], 1, -1)
        x = torch.cat([cls, patches], dim=1) + self.position.weight  # (batch, 50, 768)
        return self.ln(x)


# %% [image-encoder] Image encoder
class ImageEncoder(nn.Module):
    """12 bidirectional blocks; the [CLS] token's final vector, LayerNormed, stands for the image."""

    def __init__(self, c: Config):
        super().__init__()
        self.embedding = ImageEmbedding(c)
        self.blocks = nn.ModuleList(Block(c.vision, causal=False) for _ in range(c.vision.n_layers))
        self.ln_post = nn.LayerNorm(c.vision.d_model, eps=c.vision.eps)

    def forward(self, pixels):
        x = self.embedding(pixels)
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x[:, 0])  # pooled: (batch, 768)


# %% [text-embedding] Text embedding
class TextEmbedding(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.text.d_model)
        self.position = nn.Embedding(c.max_positions, c.text.d_model)

    def forward(self, ids):  # ids: (captions, seq), starting with <start of text>
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.token(ids) + self.position(positions)  # (captions, seq, 512)


# %% [text-encoder] Text encoder
class TextEncoder(nn.Module):
    """12 causal blocks and a final LayerNorm; the end-of-text token's vector stands for the caption."""

    def __init__(self, c: Config):
        super().__init__()
        self.embedding = TextEmbedding(c)
        self.blocks = nn.ModuleList(Block(c.text, causal=True) for _ in range(c.text.n_layers))
        self.ln_final = nn.LayerNorm(c.text.d_model, eps=c.text.eps)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        eot = ids.argmax(dim=-1)  # <end of text> has the largest id in the vocabulary, 49407
        return x[torch.arange(ids.shape[0]), eot]  # pooled: (captions, 512)


# %% [projection] Pool & project
class Projection(nn.Module):
    """Map a pooled vector into the shared space and scale it to unit length."""

    def __init__(self, d_in: int, d_shared: int):
        super().__init__()
        self.weight = nn.Linear(d_in, d_shared, bias=False)  # W_img (768 → 512) or W_txt (512 → 512)

    def forward(self, pooled):
        u = self.weight(pooled)
        return u / u.norm(dim=-1, keepdim=True)  # (batch, 512), ‖u‖ = 1


# %% [similarity] Similarity
class Similarity(nn.Module):
    """Cosine similarity of every image with every caption, times a learned temperature."""

    def __init__(self, c: Config):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(c.logit_scale))

    def forward(self, images, captions):  # unit vectors: (images, 512), (captions, 512)
        cosine = images @ captions.T  # (images, captions)
        return self.logit_scale.exp() * cosine  # logits; trained CLIP multiplies by 100


def caption_probabilities(logits):
    return logits.softmax(dim=-1)  # for each image, a distribution over the captions


# %% [model] Full model
class CLIP(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.image_encoder = ImageEncoder(c)
        self.text_encoder = TextEncoder(c)
        self.image_projection = Projection(c.vision.d_model, c.d_shared)
        self.text_projection = Projection(c.text.d_model, c.d_shared)
        self.similarity = Similarity(c)

    def forward(self, pixels, ids):
        images = self.image_projection(self.image_encoder(pixels))
        captions = self.text_projection(self.text_encoder(ids))
        return self.similarity(images, captions)  # (images, captions)


# probabilities = caption_probabilities(CLIP()(pixels, ids))  # zero-shot: which caption fits each image
`;export{e as default};
