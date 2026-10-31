const e=`# SAM 3 (facebook/sam3), the detector in the SAM 3 tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram. The image encoder runs once per image; the text
# encoder, the fusion encoder, the DETR decoder with its presence token and the mask decoder run once per
# concept ("dog", "ear", ...). Every query that scores high enough is one instance of the concept.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    image_size: int = 1008  # the photo is stretched to 1008 × 1008, not padded
    patch_size: int = 14  # a 72 × 72 grid of 5,184 patches
    channels: int = 3
    d_model: int = 1024
    n_heads: int = 16  # 64 dimensions per head
    n_layers: int = 32
    d_mlp: int = 4736
    window: int = 24  # windowed blocks attend inside 24 × 24 windows: 3 × 3 of them, no padding
    global_layers: tuple = (7, 15, 23, 31)  # blocks 8, 16, 24 and 32 attend over all 5,184 patches
    pretrain_grid: int = 24  # the learned positions cover 24 × 24 cells and are tiled 3 × 3
    rope_theta: float = 10000.0
    eps: float = 1e-6
    fpn_scales: tuple = (4.0, 2.0, 1.0, 0.5)  # 288², 144², 72² and 36² feature maps
    d: int = 256  # the detector's width: FPN channels, text tokens, queries
    text_vocab: int = 49408
    text_len: int = 32  # every concept is padded to 32 tokens
    d_text: int = 1024
    text_heads: int = 16
    text_layers: int = 24
    text_mlp: int = 4096
    text_eps: float = 1e-5
    heads: int = 8  # attention heads in the fusion encoder, the decoder and the mask decoder
    d_ffn: int = 2048
    encoder_layers: int = 6
    decoder_layers: int = 6
    n_queries: int = 200
    detector_eps: float = 1e-5  # the detector's LayerNorms use PyTorch's default


# %% [patch-embedding] Embedding
class PatchEmbedding(nn.Module):
    """Wpatch on every 14 × 14 patch, plus learned positions tiled over the grid, then a LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        self.grid, self.pretrain_grid = c.image_size // c.patch_size, c.pretrain_grid
        self.projection = nn.Conv2d(c.channels, c.d_model, kernel_size=c.patch_size, stride=c.patch_size, bias=False)
        self.position = nn.Parameter(torch.zeros(1, c.pretrain_grid**2, c.d_model))  # 24 × 24 cells
        self.ln = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, pixels):  # pixels: (batch, 3, 1008, 1008)
        x = self.projection(pixels).permute(0, 2, 3, 1)  # the grid stays 2-D: (batch, 72, 72, 1024)
        g, p = self.grid, self.pretrain_grid
        # The 24 × 24 table repeats 3 × 3 times: patch (r, c) gets the position of (r mod 24, c mod 24).
        position = self.position.view(p, p, -1).repeat(-(-g // p), -(-g // p), 1)[:g, :g]
        return self.ln(x + position)


# %% [rope] 2-D rotary positions
def rope_angles(c: Config, size: int):
    """One rotation angle per pair of query/key coordinates and grid cell: (size², 64).

    Windowed blocks number the cells of a 24 × 24 window 0 … 23. Global blocks squeeze the 72 × 72 grid
    into the same range (steps of 1/3), so both see positions like the ones they were trained on."""
    head_dim = c.d_model // c.n_heads
    inv_freq = 1.0 / c.rope_theta ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))  # 16
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    scale = c.window / size
    x, y = x.flatten() * scale, y.flatten() * scale
    angles = torch.cat([x[:, None] * inv_freq, y[:, None] * inv_freq], dim=-1)  # 16 for x, then 16 for y
    return angles.repeat_interleave(2, dim=-1)  # each angle turns one adjacent pair of coordinates


def rotate(x, angles):  # x: (..., cells, 64)
    """Turn each pair (x₂ᵢ, x₂ᵢ₊₁) by its angle."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    turned = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * angles.cos() + turned * angles.sin()


# %% [attention] Windowed / global self-attention
class Attention(nn.Module):
    """Multi-head self-attention over a 2-D grid. Q and K are rotated by their cell's position first."""

    def __init__(self, c: Config, size: int):  # size: 24 for a window, 72 for the whole grid
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)
        self.register_buffer("angles", rope_angles(c, size), persistent=False)

    def forward(self, x):  # x: (windows or batch, h, w, d_model)
        b, h, w, d = x.shape
        split = lambda t: t.view(b, h * w, self.n_heads, -1).transpose(1, 2)  # (b, heads, h·w, 64)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        q, k = rotate(q, self.angles), rotate(k, self.angles)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (b, heads, h·w, h·w)
        return self.out((weights @ v).transpose(1, 2).reshape(b, h, w, d))


# %% [mlp] MLP
class MLP(nn.Module):
    """GELU in the image and text encoders, ReLU in the detector (256 → 2048 → 256)."""

    def __init__(self, d: int, d_mlp: int, activation=F.gelu):
        super().__init__()
        self.up = nn.Linear(d, d_mlp)
        self.down = nn.Linear(d_mlp, d)
        self.activation = activation

    def forward(self, x):
        return self.down(self.activation(self.up(x)))


# %% [block] Encoder block
def to_windows(x, size):
    """(batch, H, W, d) → (batch · windows, size, size, d); 72 × 72 splits into 3 × 3 windows exactly."""
    b, H, W, d = x.shape
    x = F.pad(x, (0, 0, 0, -W % size, 0, -H % size))
    rows, cols = x.shape[1] // size, x.shape[2] // size
    x = x.view(b, rows, size, cols, size, d).transpose(2, 3)
    return x.reshape(b * rows * cols, size, size, d)


def from_windows(windows, size, H, W):
    rows, cols = -(-H // size), -(-W // size)
    b = windows.shape[0] // (rows * cols)
    x = windows.view(b, rows, cols, size, size, -1).transpose(2, 3).reshape(b, rows * size, cols * size, -1)
    return x[:, :H, :W]


class Block(nn.Module):
    """Pre-norm residual block. Windowed blocks attend inside each 24 × 24 window only."""

    def __init__(self, c: Config, window: int):  # window = 0 for a global block
        super().__init__()
        self.window = window
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c, size=window or c.image_size // c.patch_size)
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c.d_model, c.d_mlp)

    def forward(self, x):  # x: (batch, 72, 72, 1024)
        h = self.ln_1(x)
        if self.window:
            h = from_windows(self.attention(to_windows(h, self.window)), self.window, x.shape[1], x.shape[2])
        else:
            h = self.attention(h)
        x = x + h
        return x + self.mlp(self.ln_2(x))


# %% [neck] FPN neck
def sine_positions(d: int, h: int, w: int):
    """DETR's fixed 2-D sine positions: d / 2 channels for the row, d / 2 for the column, (d, h, w)."""
    y = torch.arange(1, h + 1).float()[:, None].expand(h, w) / (h + 1e-6) * (2 * math.pi)
    x = torch.arange(1, w + 1).float()[None, :].expand(h, w) / (w + 1e-6) * (2 * math.pi)
    half = d // 2
    dim_t = 10000 ** (2 * (torch.arange(half) // 2) / half)

    def encode(v):  # sin and cos interleaved: channel 2i is sin(v / dᵢ), 2i + 1 is cos(v / dᵢ)
        v = v[..., None] / dim_t
        return torch.stack([v[..., 0::2].sin(), v[..., 1::2].cos()], dim=-1).flatten(-2)

    return torch.cat([encode(y), encode(x)], dim=-1).permute(2, 0, 1)


class Scale(nn.Module):
    """One pyramid level from the 72 × 72 × 1024 grid: resample, then a 1 × 1 and a 3 × 3 convolution."""

    def __init__(self, c: Config, scale: float):
        super().__init__()
        d = c.d_model
        if scale == 4.0:  # two transposed convolutions: 72 → 144 → 288
            self.resample = nn.Sequential(
                nn.ConvTranspose2d(d, d // 2, 2, 2), nn.GELU(), nn.ConvTranspose2d(d // 2, d // 4, 2, 2)
            )
            channels = d // 4
        elif scale == 2.0:  # 72 → 144
            self.resample, channels = nn.Sequential(nn.ConvTranspose2d(d, d // 2, 2, 2)), d // 2
        elif scale == 1.0:
            self.resample, channels = nn.Sequential(), d
        else:  # 72 → 36
            self.resample, channels = nn.Sequential(nn.MaxPool2d(2, 2)), d
        self.proj1 = nn.Conv2d(channels, c.d, kernel_size=1)
        self.proj2 = nn.Conv2d(c.d, c.d, kernel_size=3, padding=1)

    def forward(self, x):  # x: (batch, 1024, 72, 72)
        return self.proj2(self.proj1(self.resample(x)))


class Neck(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.scales = nn.ModuleList(Scale(c, s) for s in c.fpn_scales)

    def forward(self, x):  # x: (batch, 72, 72, 1024)
        x = x.permute(0, 3, 1, 2)
        levels = [scale(x) for scale in self.scales]  # (batch, 256, 288, 288) … (batch, 256, 36, 36)
        return levels, [sine_positions(level.shape[1], *level.shape[-2:]) for level in levels]


# %% [image-encoder] Image encoder
class ImageEncoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embedding = PatchEmbedding(c)
        self.blocks = nn.ModuleList(Block(c, window=0 if i in c.global_layers else c.window) for i in range(c.n_layers))
        self.neck = Neck(c)

    def forward(self, pixels):
        x = self.embedding(pixels)
        for block in self.blocks:
            x = block(x)
        return self.neck(x)  # four levels and their positions, computed once and reused for every concept


# %% [text-encoder] Text encoder
class TextBlock(nn.Module):
    """CLIP's pre-norm block: causal self-attention, then a GELU MLP (1024 → 4096 → 1024)."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.text_heads
        self.ln_1 = nn.LayerNorm(c.d_text, eps=c.text_eps)
        self.q, self.k, self.v, self.out = (nn.Linear(c.d_text, c.d_text) for _ in range(4))
        self.ln_2 = nn.LayerNorm(c.d_text, eps=c.text_eps)
        self.mlp = MLP(c.d_text, c.text_mlp)

    def forward(self, x, mask):  # mask: (batch, 1, tokens, tokens), 0 or −∞
        b, t, d = x.shape
        split = lambda z: z.view(b, t, self.n_heads, -1).transpose(1, 2)
        h = self.ln_1(x)
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]) + mask).softmax(dim=-1)
        x = x + self.out((weights @ v).transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln_2(x))


class TextEncoder(nn.Module):
    """The concept's tokens through CLIP's 24-block text transformer, every token projected to 256."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.text_vocab, c.d_text)
        self.position = nn.Parameter(torch.zeros(c.text_len, c.d_text))
        self.blocks = nn.ModuleList(TextBlock(c) for _ in range(c.text_layers))
        self.ln_final = nn.LayerNorm(c.d_text, eps=c.text_eps)
        self.projection = nn.Linear(c.d_text, c.d)

    def forward(self, ids, valid):  # ids, valid: (batch, 32); valid is False on the padding
        t = ids.shape[1]
        causal = torch.full((t, t), float("-inf")).triu(1)  # each token sees itself and the tokens before it
        mask = causal + torch.where(valid, 0.0, float("-inf"))[:, None, None, :]  # and never the padding
        x = self.token(ids) + self.position[:t]
        for block in self.blocks:
            x = block(x, mask)
        return self.projection(self.ln_final(x))  # (batch, 32, 256): every token, not one pooled vector


# %% [fusion-encoder] Fusion encoder
class DetectorAttention(nn.Module):
    """Multi-head attention with separate query, key and value inputs and an optional additive bias."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.heads
        self.q, self.k, self.v, self.out = (nn.Linear(c.d, c.d) for _ in range(4))

    def forward(self, query, key, value, bias=None):  # bias: (batch, heads or 1, n_q, n_k)
        split = lambda z: z.view(z.shape[0], z.shape[1], self.n_heads, -1).transpose(1, 2)
        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(value))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if bias is not None:
            scores = scores + bias
        heads = scores.softmax(dim=-1) @ v
        return self.out(heads.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


def text_bias(valid):
    """−∞ on the padded text tokens as keys: (batch, 1, 1, 32)."""
    return torch.where(valid, 0.0, float("-inf"))[:, None, None, :]


class FusionLayer(nn.Module):
    """Pre-norm: the image attends to itself, then to the concept's tokens, then an MLP."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.self_attention = DetectorAttention(c)
        self.ln_2 = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.text_attention = DetectorAttention(c)
        self.ln_3 = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.mlp = MLP(c.d, c.d_ffn, activation=F.relu)

    def forward(self, image, image_pos, text, valid):  # image: (batch, 5184, 256), text: (batch, 32, 256)
        h = self.ln_1(image)
        image = image + self.self_attention(h + image_pos, h + image_pos, h)  # positions on Q and K only
        image = image + self.text_attention(self.ln_2(image), text, text, text_bias(valid))
        return image + self.mlp(self.ln_3(image))


class FusionEncoder(nn.Module):
    """Six layers over the 72 × 72 level: the image features become features of this concept."""

    def __init__(self, c: Config):
        super().__init__()
        self.layers = nn.ModuleList(FusionLayer(c) for _ in range(c.encoder_layers))

    def forward(self, level, level_pos, text, valid):  # level: (batch, 256, 72, 72)
        image = level.flatten(2).transpose(1, 2)  # (batch, 5184, 256)
        image_pos = level_pos.flatten(1).T  # (5184, 256)
        for layer in self.layers:
            image = layer(image, image_pos, text, valid)
        return image, image_pos


# %% [box-bias] Box position bias
class FeedForward(nn.Module):
    """Linear layers with ReLU between them (none after the last)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, layers: int):
        super().__init__()
        sizes = [d_in] + [d_hidden] * (layers - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x) if i == len(self.layers) - 1 else F.relu(layer(x))
        return x


def corners(box):  # (cx, cy, w, h) → (x₀, y₀, x₁, y₁)
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class BoxBias(nn.Module):
    """An attention bias for each query and image cell from where the cell is relative to the query's box.

    A cell's signed distance to the box's left and right edges (and to its top and bottom) is squashed
    with a log, then two small MLPs turn it into one number per head."""

    def __init__(self, c: Config):
        super().__init__()
        self.mlp_x = FeedForward(2, c.d, c.heads, 2)
        self.mlp_y = FeedForward(2, c.d, c.heads, 2)

    def forward(self, boxes, h: int, w: int):  # boxes: (batch, queries, 4) as (cx, cy, w, h) in 0 … 1
        x0, y0, x1, y1 = corners(boxes).unbind(-1)
        ys, xs = torch.arange(h).float() / h, torch.arange(w).float() / w  # each cell's top-left corner
        dx = xs[None, None, :, None] - torch.stack([x0, x1], dim=-1)[:, :, None, :]  # (batch, queries, w, 2)
        dy = ys[None, None, :, None] - torch.stack([y0, y1], dim=-1)[:, :, None, :]  # (batch, queries, h, 2)
        squash = lambda d: torch.sign(d * 8) * torch.log2((d * 8).abs() + 1) / 3  # log₂(1 + 8|d|) / log₂ 8
        bias = self.mlp_y(squash(dy))[:, :, :, None] + self.mlp_x(squash(dx))[:, :, None, :]  # (…, h, w, heads)
        return bias.flatten(2, 3).permute(0, 3, 1, 2)  # (batch, heads, queries, h · w)


# %% [decoder] DETR decoder
def logit(p, eps=1e-3):
    """The inverse of the sigmoid, with p and 1 − p kept at least eps."""
    return torch.log(p.clamp(min=eps) / (1 - p).clamp(min=eps))


def box_positions(boxes, d: int):
    """Sine encodings of a box's centre and size, 4 × d / 2 numbers: (y, x, w, h)."""
    dim_t = 10000 ** (2 * (torch.arange(d // 2) // 2) / (d // 2))

    def encode(v):
        v = v[..., None] * (2 * math.pi) / dim_t
        return torch.stack([v[..., 0::2].sin(), v[..., 1::2].cos()], dim=-1).flatten(-2)

    cx, cy, w, h = boxes.unbind(-1)
    return torch.cat([encode(cy), encode(cx), encode(w), encode(h)], dim=-1)


class DecoderLayer(nn.Module):
    """Post-norm: the queries attend to each other, to the concept's tokens and to the image, then an MLP."""

    def __init__(self, c: Config):
        super().__init__()
        self.self_attention = DetectorAttention(c)
        self.ln_1 = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.text_attention = DetectorAttention(c)
        self.ln_2 = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.image_attention = DetectorAttention(c)
        self.ln_3 = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.mlp = MLP(c.d, c.d_ffn, activation=F.relu)
        self.ln_4 = nn.LayerNorm(c.d, eps=c.detector_eps)

    def forward(self, x, query_pos, text, valid, image, image_pos, box_bias):
        q = x + query_pos
        x = self.ln_1(x + self.self_attention(q, q, x))
        x = self.ln_2(x + self.text_attention(x + query_pos, text, text, text_bias(valid)))
        x = self.ln_3(x + self.image_attention(x + query_pos, image + image_pos, image, box_bias))
        return self.ln_4(x + self.mlp(x))


class Decoder(nn.Module):
    """200 learned queries, each with a box it refines layer by layer, and one presence token in front."""

    def __init__(self, c: Config):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(c.n_queries, c.d))
        self.reference_boxes = nn.Parameter(torch.zeros(c.n_queries, 4))  # before the sigmoid
        self.presence_token = nn.Parameter(torch.zeros(1, c.d))
        self.query_pos = FeedForward(2 * c.d, c.d, c.d, 2)  # box sine encoding → the query's position
        self.box_bias = BoxBias(c)
        self.layers = nn.ModuleList(DecoderLayer(c) for _ in range(c.decoder_layers))
        self.ln_out = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.box_head = FeedForward(c.d, c.d, 4, 3)  # a change of the box, before the sigmoid
        self.presence = PresenceHead(c)

    def forward(self, text, valid, image, image_pos, grid: int):
        b = text.shape[0]
        x = torch.cat([self.presence_token, self.queries]).expand(b, -1, -1)  # (batch, 1 + 200, 256)
        boxes = self.reference_boxes.sigmoid().expand(b, -1, -1)
        states, all_boxes, presence = [], [boxes], []
        for layer in self.layers:
            query_pos = F.pad(self.query_pos(box_positions(boxes, x.shape[-1])), (0, 0, 1, 0))  # presence: 0
            bias = F.pad(self.box_bias(boxes, grid, grid), (0, 0, 1, 0))  # presence: no bias, sees everything
            x = layer(x, query_pos, text, valid, image, image_pos, bias)
            out = self.ln_out(x[:, 1:])
            boxes = (logit(boxes) + self.box_head(out)).sigmoid()  # refine each box
            states.append(out)
            all_boxes.append(boxes)
            presence.append(self.presence(x[:, :1]))
        # Per layer: query states, the boxes each layer started from (and the last layer's result), presence.
        return torch.stack(states), torch.stack(all_boxes), torch.stack(presence)


# %% [presence] Presence head
class PresenceHead(nn.Module):
    """One logit for the whole image: is the concept here at all? Clamped to ±10."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.mlp = FeedForward(c.d, c.d, 1, 3)

    def forward(self, token):  # token: (batch, 1, 256), the presence token after a decoder layer
        return self.mlp(self.ln(token)).squeeze(-1).clamp(-10, 10)  # (batch, 1)


# %% [scoring] Query scoring
class Scoring(nn.Module):
    """Each query's logit: its projection · the projection of the concept's mean token, ÷ √256, clamped ±12."""

    def __init__(self, c: Config):
        super().__init__()
        self.text_mlp = MLP(c.d, c.d_ffn, activation=F.relu)
        self.ln = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.text_proj = nn.Linear(c.d, c.d)
        self.query_proj = nn.Linear(c.d, c.d)

    def forward(self, queries, text, valid):  # queries: (…, batch, 200, 256)
        text = self.ln(text + self.text_mlp(text))
        keep = valid[..., None].float()
        pooled = (text * keep).sum(1) / keep.sum(1).clamp(min=1)  # mean over the real tokens: (batch, 256)
        logits = self.query_proj(queries) @ self.text_proj(pooled)[..., None] / math.sqrt(text.shape[-1])
        return logits.squeeze(-1).clamp(-12, 12)  # (…, batch, 200)


# %% [mask-decoder] Mask decoder
class PixelDecoder(nn.Module):
    """From the fused 72 × 72 map up to 288 × 288: upsample, add the FPN level, 3 × 3 conv, GroupNorm, ReLU."""

    def __init__(self, c: Config):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(c.d, c.d, 3, padding=1) for _ in range(3))
        self.norms = nn.ModuleList(nn.GroupNorm(8, c.d) for _ in range(3))

    def forward(self, levels):  # levels: [288², 144², 72²]; the last is the fusion encoder's output
        x = levels[-1]
        for i, skip in enumerate(reversed(levels[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest") + skip
            x = F.relu(self.norms[i](self.convs[i](x)))
        return x  # (batch, 256, 288, 288)


class MaskDecoder(nn.Module):
    """Each query becomes a mask: its embedding · every pixel's embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln = nn.LayerNorm(c.d, eps=c.detector_eps)
        self.text_attention = DetectorAttention(c)  # the fused image reads the concept once more
        self.pixel_decoder = PixelDecoder(c)
        self.pixel_proj = nn.Conv2d(c.d, c.d, kernel_size=1)
        self.mask_mlp = FeedForward(c.d, c.d, c.d, 3)
        self.semantic = nn.Conv2d(c.d, 1, kernel_size=1)  # one "concept anywhere" map, not used for instances

    def forward(self, queries, levels, image, text, valid):  # queries: (batch, 200, 256), image: (batch, 5184, 256)
        image = image + self.text_attention(self.ln(image), text, text, text_bias(valid))
        g = levels[2].shape[-1]
        fused = image.transpose(1, 2).reshape(image.shape[0], -1, g, g)
        pixels = self.pixel_decoder([levels[0], levels[1], fused])
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_mlp(queries), self.pixel_proj(pixels))
        return masks, self.semantic(pixels)  # (batch, 200, 288, 288) mask logits, (batch, 1, 288, 288)


# %% [detections] Detections
def detections(logits, presence, boxes, masks, size, threshold=0.5):
    """Score = sigmoid(query logit) × sigmoid(presence logit). Keep the queries above the threshold, resize
    their masks to the photo's size (height, width) and binarize at 0.5; boxes go from 0 … 1 to pixels."""
    scores = logits.sigmoid() * presence.sigmoid()  # (batch, 200) × (batch, 1)
    results = []
    for s, b, m in zip(scores, corners(boxes), masks):
        keep = s > threshold
        probability = m[keep][None].sigmoid()  # (1, kept, 288, 288)
        if keep.any():  # an absent concept keeps nothing
            probability = F.interpolate(probability, size, mode="bilinear", align_corners=False)
        else:
            probability = probability.new_zeros(1, 0, *size)
        pixels = b[keep] * torch.tensor([size[1], size[0], size[1], size[0]])
        results.append(dict(scores=s[keep], boxes=pixels, masks=probability[0] > 0.5))
    return results


# %% [model] Full model
class SAM3(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.image_encoder = ImageEncoder(c)
        self.text_encoder = TextEncoder(c)
        self.fusion_encoder = FusionEncoder(c)
        self.decoder = Decoder(c)
        self.scoring = Scoring(c)
        self.mask_decoder = MaskDecoder(c)

    def forward(self, pixels, ids, valid):
        levels, positions = self.image_encoder(pixels)  # once per image
        text = self.text_encoder(ids, valid)  # once per concept, from here on
        image, image_pos = self.fusion_encoder(levels[2], positions[2], text, valid)
        states, boxes, presence = self.decoder(text, valid, image, image_pos, levels[2].shape[-1])
        logits = self.scoring(states, text, valid)  # (layers, batch, 200)
        masks, semantic = self.mask_decoder(states[-1], levels[:3], image, text, valid)
        return logits[-1], presence[-1], boxes[-1], masks  # the last layer's answers


# logits, presence, boxes, masks = SAM3()(pixels, ids, valid)
# found = detections(logits, presence, boxes, masks, size=(750, 1000))  # every instance of the concept
`;export{e as default};
