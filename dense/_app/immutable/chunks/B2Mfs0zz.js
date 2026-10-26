const e=`# SAM ViT-B (facebook/sam-vit-base), the model in the SAM tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram. The image encoder runs once per image; the
# prompt encoder and mask decoder run once per prompt (a click or a box) on the same image embedding.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    image_size: int = 1024  # the photo is resized so its long side is 1024, then padded to a square
    patch_size: int = 16  # a 64 × 64 grid of 4,096 patches
    channels: int = 3
    d_model: int = 768
    n_heads: int = 12  # 64 dimensions per head
    n_layers: int = 12
    d_mlp: int = 3072
    window: int = 14  # windowed blocks attend inside 14 × 14 windows of the grid (padded to 70 × 70)
    global_layers: tuple = (2, 5, 8, 11)  # blocks 3, 6, 9 and 12 attend over all 4,096 patches
    eps: float = 1e-6
    d_embed: int = 256  # channels of the image embedding, the prompt tokens and the decoder
    n_fourier: int = 128  # random Fourier features per coordinate: sin and cos give 256 numbers
    decoder_heads: int = 8
    decoder_layers: int = 2
    decoder_mlp: int = 2048
    downsample: int = 2  # token ↔ image attention works in 256 / 2 = 128 dimensions
    n_mask_tokens: int = 4  # one single-mask answer + three multimask answers
    d_hyper: int = 32  # 256 / 8 channels in the upscaled image embedding


# %% [patch-embedding] Embedding
class PatchEmbedding(nn.Module):
    """Wpatch on every 16 × 16 patch, plus a learned absolute position for each grid cell."""

    def __init__(self, c: Config):
        super().__init__()
        grid = c.image_size // c.patch_size
        # A 16 × 16 convolution with stride 16 flattens each patch to 768 numbers and multiplies by Wpatch.
        self.projection = nn.Conv2d(c.channels, c.d_model, kernel_size=c.patch_size, stride=c.patch_size)
        self.position = nn.Parameter(torch.zeros(1, grid, grid, c.d_model))

    def forward(self, pixels):  # pixels: (batch, 3, 1024, 1024)
        x = self.projection(pixels).permute(0, 2, 3, 1)  # the grid stays 2-D: (batch, 64, 64, 768)
        return x + self.position


# %% [attention] Windowed / global self-attention
def relative_positions(table, size):
    """The table row for every (query, key) offset along one axis: (size, size, head_dim)."""
    offsets = torch.arange(size)[:, None] - torch.arange(size)[None, :] + size - 1  # 0 … 2·size − 2
    return table[offsets]


class Attention(nn.Module):
    """Self-attention over a 2-D grid, with learned relative positions added along rows and columns."""

    def __init__(self, c: Config, size: int):  # size: 14 for a window, 64 for the whole grid
        super().__init__()
        self.n_heads = c.n_heads
        head_dim = c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * size - 1, head_dim))  # one row per vertical offset
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * size - 1, head_dim))  # one row per horizontal offset

    def forward(self, x):  # x: (windows or batch, h, w, d_model)
        b, h, w, d = x.shape
        # Q, K and V for every head at once, each (b, heads, h·w, head_dim)
        q, k, v = self.qkv(x).view(b, h * w, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (b, heads, h·w, h·w)
        # Decomposed relative position: query (y, x) and key (y', x') add q·Rh[y − y'] + q·Rw[x − x'].
        grid_q = q.reshape(b, self.n_heads, h, w, -1)
        rows = torch.einsum("bnyxc,ykc->bnyxk", grid_q, relative_positions(self.rel_pos_h, h))  # (…, h, w, h)
        cols = torch.einsum("bnyxc,xkc->bnyxk", grid_q, relative_positions(self.rel_pos_w, w))  # (…, h, w, w)
        scores = scores + (rows[..., :, None] + cols[..., None, :]).reshape(b, self.n_heads, h * w, h * w)
        heads = scores.softmax(dim=-1) @ v  # (b, heads, h·w, head_dim)
        return self.out(heads.transpose(1, 2).reshape(b, h, w, d))


# %% [mlp] MLP
class MLP(nn.Module):
    """GELU in the image encoder (768 → 3072 → 768), ReLU in the mask decoder (256 → 2048 → 256)."""

    def __init__(self, d: int, d_mlp: int, activation=F.gelu):
        super().__init__()
        self.up = nn.Linear(d, d_mlp)
        self.down = nn.Linear(d_mlp, d)
        self.activation = activation

    def forward(self, x):
        return self.down(self.activation(self.up(x)))


# %% [block] Encoder block
def to_windows(x, size):
    """(batch, H, W, d) → (batch · windows, size, size, d), padding H and W up to a multiple of size."""
    b, H, W, d = x.shape
    x = F.pad(x, (0, 0, 0, -W % size, 0, -H % size))  # 64 × 64 → 70 × 70 with zeros
    rows, cols = x.shape[1] // size, x.shape[2] // size
    x = x.view(b, rows, size, cols, size, d).transpose(2, 3)
    return x.reshape(b * rows * cols, size, size, d)


def from_windows(windows, size, H, W):
    """Put the windows back in place and drop the padding: (batch, H, W, d)."""
    rows, cols = -(-H // size), -(-W // size)
    b = windows.shape[0] // (rows * cols)
    x = windows.view(b, rows, cols, size, size, -1).transpose(2, 3).reshape(b, rows * size, cols * size, -1)
    return x[:, :H, :W]


class Block(nn.Module):
    """Pre-norm residual block. Windowed blocks attend inside each 14 × 14 window only."""

    def __init__(self, c: Config, window: int):  # window = 0 for a global block
        super().__init__()
        self.window = window
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c, size=window or c.image_size // c.patch_size)
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c.d_model, c.d_mlp)

    def forward(self, x):  # x: (batch, 64, 64, 768)
        h = self.ln_1(x)
        if self.window:
            h = from_windows(self.attention(to_windows(h, self.window)), self.window, x.shape[1], x.shape[2])
        else:
            h = self.attention(h)
        x = x + h
        return x + self.mlp(self.ln_2(x))


# %% [neck] Neck
class ChannelNorm(nn.LayerNorm):
    """LayerNorm over the channels of a (batch, channels, height, width) map."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Neck(nn.Module):
    """768 → 256 channels: a 1 × 1 convolution, LayerNorm, a 3 × 3 convolution, LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        self.conv1 = nn.Conv2d(c.d_model, c.d_embed, kernel_size=1, bias=False)
        self.ln_1 = ChannelNorm(c.d_embed, eps=c.eps)
        self.conv2 = nn.Conv2d(c.d_embed, c.d_embed, kernel_size=3, padding=1, bias=False)
        self.ln_2 = ChannelNorm(c.d_embed, eps=c.eps)

    def forward(self, x):  # x: (batch, 64, 64, 768)
        x = self.ln_1(self.conv1(x.permute(0, 3, 1, 2)))
        return self.ln_2(self.conv2(x))  # image embedding: (batch, 256, 64, 64)


# %% [image-encoder] Image encoder
class ImageEncoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embedding = PatchEmbedding(c)
        self.blocks = nn.ModuleList(
            Block(c, window=0 if i in c.global_layers else c.window) for i in range(c.n_layers)
        )
        self.neck = Neck(c)

    def forward(self, pixels):
        x = self.embedding(pixels)
        for block in self.blocks:
            x = block(x)
        return self.neck(x)  # (batch, 256, 64, 64), computed once and reused for every prompt


# %% [prompt-encoder] Prompt encoder
class FourierPositions(nn.Module):
    """Random Fourier features: a fixed Gaussian matrix maps (x, y) in [0, 1] to 256 numbers."""

    def __init__(self, c: Config):
        super().__init__()
        self.register_buffer("gaussian", torch.randn(2, c.n_fourier))  # not trained

    def forward(self, xy):  # xy: (..., 2)
        angles = 2 * math.pi * ((2 * xy - 1) @ self.gaussian)  # (..., 128)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (..., 256)


class PromptEncoder(nn.Module):
    """Each click or box corner becomes one 256-D token: its Fourier position plus a learned type vector."""

    def __init__(self, c: Config):
        super().__init__()
        self.image_size, self.grid = c.image_size, c.image_size // c.patch_size
        self.positions = FourierPositions(c)
        # Learned type vectors: background click, foreground click, box top-left, box bottom-right.
        self.point_type = nn.ParameterList(nn.Parameter(torch.zeros(1, c.d_embed)) for _ in range(4))
        self.not_a_point = nn.Parameter(torch.zeros(1, c.d_embed))  # pads a click prompt without a box
        self.no_mask = nn.Parameter(torch.zeros(1, c.d_embed))  # added to every image cell (no mask prompt)

    def encode(self, xy):  # xy: pixel coordinates in the 1024 × 1024 input
        return self.positions((xy + 0.5) / self.image_size)  # + 0.5: the pixel's centre

    def forward(self, points=None, labels=None, box=None):
        """points: (batch, n, 2) with labels (batch, n), 1 = foreground, 0 = background; box: (batch, 4)."""
        tokens = []
        if points is not None:
            if box is None:  # a click prompt ends with one padding point
                points = torch.cat([points, torch.zeros_like(points[:, :1])], dim=1)
                labels = torch.cat([labels, -torch.ones_like(labels[:, :1])], dim=1)
            clicks = self.encode(points)
            clicks = torch.where(labels[..., None] == -1, self.not_a_point, clicks)
            clicks = clicks + torch.where(labels[..., None] == 0, self.point_type[0], 0)
            clicks = clicks + torch.where(labels[..., None] == 1, self.point_type[1], 0)
            tokens.append(clicks)
        if box is not None:
            corners = self.encode(box.view(-1, 2, 2))  # (batch, 2 corners, 256)
            tokens.append(corners + torch.cat([self.point_type[2], self.point_type[3]]))
        sparse = torch.cat(tokens, dim=1)  # (batch, prompt tokens, 256)
        dense = self.no_mask.view(1, -1, 1, 1).expand(sparse.shape[0], -1, self.grid, self.grid)
        return sparse, dense

    def image_positions(self):
        """The Fourier position of every cell centre of the 64 × 64 image embedding: (256, 64, 64)."""
        centres = (torch.arange(self.grid, device=self.no_mask.device) + 0.5) / self.grid
        y, x = torch.meshgrid(centres, centres, indexing="ij")
        return self.positions(torch.stack([x, y], dim=-1)).permute(2, 0, 1)


# %% [two-way-attention] Two-way attention
class DecoderAttention(nn.Module):
    """Attention with separate query, key and value inputs, in d / downsample dimensions."""

    def __init__(self, c: Config, downsample: int):
        super().__init__()
        self.n_heads = c.decoder_heads
        inner = c.d_embed // downsample
        self.q = nn.Linear(c.d_embed, inner)
        self.k = nn.Linear(c.d_embed, inner)
        self.v = nn.Linear(c.d_embed, inner)
        self.out = nn.Linear(inner, c.d_embed)

    def split(self, x):  # (batch, n, inner) → (batch, heads, n, inner / heads)
        return x.view(x.shape[0], x.shape[1], self.n_heads, -1).transpose(1, 2)

    def forward(self, query, key, value):  # (batch, n_q, 256), (batch, n_k, 256), (batch, n_k, 256)
        q, k, v = self.split(self.q(query)), self.split(self.k(key)), self.split(self.v(value))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (batch, heads, n_q, n_k)
        heads = weights @ v
        return self.out(heads.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


class TwoWayBlock(nn.Module):
    """Tokens attend to themselves and to the image, an MLP, then the image attends back to the tokens."""

    def __init__(self, c: Config, first: bool):
        super().__init__()
        self.first = first
        self.self_attention = DecoderAttention(c, downsample=1)
        self.ln_1 = nn.LayerNorm(c.d_embed, eps=c.eps)
        self.token_to_image = DecoderAttention(c, downsample=c.downsample)
        self.ln_2 = nn.LayerNorm(c.d_embed, eps=c.eps)
        self.mlp = MLP(c.d_embed, c.decoder_mlp, activation=F.relu)
        self.ln_3 = nn.LayerNorm(c.d_embed, eps=c.eps)
        self.image_to_token = DecoderAttention(c, downsample=c.downsample)
        self.ln_4 = nn.LayerNorm(c.d_embed, eps=c.eps)

    def forward(self, tokens, image, token_pe, image_pe):  # the prompt tokens are their own positions
        if self.first:  # layer 1 replaces the tokens with their self-attention: no positions, no residual
            tokens = self.self_attention(tokens, tokens, tokens)
        else:
            q = tokens + token_pe
            tokens = tokens + self.self_attention(q, q, tokens)
        tokens = self.ln_1(tokens)
        tokens = self.ln_2(tokens + self.token_to_image(tokens + token_pe, image + image_pe, image))
        tokens = self.ln_3(tokens + self.mlp(tokens))
        image = self.ln_4(image + self.image_to_token(image + image_pe, tokens + token_pe, tokens))
        return tokens, image


class TwoWayTransformer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.layers = nn.ModuleList(TwoWayBlock(c, first=i == 0) for i in range(c.decoder_layers))
        self.final_token_to_image = DecoderAttention(c, downsample=c.downsample)
        self.ln_final = nn.LayerNorm(c.d_embed)  # PyTorch's default eps, 1e-5, as in SAM

    def forward(self, tokens, image, image_pe):  # image, image_pe: (batch, 256, 64, 64), (256, 64, 64)
        image = image.flatten(2).transpose(1, 2)  # (batch, 4096, 256)
        image_pe = image_pe.flatten(1).T  # (4096, 256)
        queries, keys = tokens, image
        for layer in self.layers:
            queries, keys = layer(queries, keys, tokens, image_pe)
        queries = queries + self.final_token_to_image(queries + tokens, keys + image_pe, keys)
        return self.ln_final(queries), keys


# %% [mask-decoder] Mask decoder
class FeedForward(nn.Module):
    """Three linear layers with ReLU between them."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class MaskDecoder(nn.Module):
    """IoU and mask output tokens go in front of the prompt tokens; each mask token becomes a mask."""

    def __init__(self, c: Config):
        super().__init__()
        d = c.d_embed
        self.iou_token = nn.Parameter(torch.zeros(1, d))
        self.mask_tokens = nn.Parameter(torch.zeros(c.n_mask_tokens, d))
        self.transformer = TwoWayTransformer(c)
        # Upscale the image side ×4: 64 × 64 × 256 → 256 × 256 × 32.
        self.upscale_1 = nn.ConvTranspose2d(d, d // 4, kernel_size=2, stride=2)
        self.upscale_norm = ChannelNorm(d // 4, eps=c.eps)
        self.upscale_2 = nn.ConvTranspose2d(d // 4, c.d_hyper, kernel_size=2, stride=2)
        # A hypernetwork per mask token turns the token into the 32 weights of its mask.
        self.hypernetworks = nn.ModuleList(FeedForward(d, d, c.d_hyper) for _ in range(c.n_mask_tokens))
        self.iou_head = FeedForward(d, d, c.n_mask_tokens)  # predicted IoU of each mask

    def forward(self, image, image_pe, sparse, dense):
        b, n_masks = image.shape[0], self.mask_tokens.shape[0]
        outputs = torch.cat([self.iou_token, self.mask_tokens]).expand(b, -1, -1)
        tokens = torch.cat([outputs, sparse], dim=1)  # (batch, 1 + 4 + prompt tokens, 256)
        tokens, image = self.transformer(tokens, image + dense, image_pe)
        image = image.transpose(1, 2).reshape(b, -1, *dense.shape[-2:])  # back to (batch, 256, 64, 64)
        upscaled = F.gelu(self.upscale_norm(self.upscale_1(image)))
        upscaled = F.gelu(self.upscale_2(upscaled))  # (batch, 32, 256, 256)
        weights = torch.stack([mlp(tokens[:, 1 + i]) for i, mlp in enumerate(self.hypernetworks)], dim=1)
        masks = (weights @ upscaled.flatten(2)).view(b, n_masks, *upscaled.shape[-2:])  # (batch, 4, 256, 256)
        iou = self.iou_head(tokens[:, 0])  # (batch, 4)
        # Mask 0 is the single-mask answer; masks 1–3 are the three multimask answers.
        return masks, iou


# %% [masks] Masks
def post_process(low_res, resized, original, image_size=1024):
    """256 × 256 mask logits → the photo's own size, then > 0. resized: the photo's size inside 1024."""
    masks = F.interpolate(low_res, (image_size, image_size), mode="bilinear", align_corners=False)
    masks = masks[..., : resized[0], : resized[1]]  # drop the padding below and right of the photo
    logits = F.interpolate(masks, original, mode="bilinear", align_corners=False)
    return logits, logits > 0


# %% [model] Full model
class SAM(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.image_encoder = ImageEncoder(c)
        self.prompt_encoder = PromptEncoder(c)
        self.mask_decoder = MaskDecoder(c)

    def forward(self, pixels, points=None, labels=None, box=None):
        image = self.image_encoder(pixels)  # once per image
        sparse, dense = self.prompt_encoder(points, labels, box)  # once per prompt
        return self.mask_decoder(image, self.prompt_encoder.image_positions(), sparse, dense)


# masks, iou = SAM()(pixels, points=torch.tensor([[[500.0, 375.0]]]), labels=torch.tensor([[1]]))
# best = 1 + iou[:, 1:].argmax(-1)  # the multimask answer SAM rates highest
`;export{e as default};
