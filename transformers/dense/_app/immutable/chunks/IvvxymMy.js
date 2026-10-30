const n=`# Emu3's vision tokenizer (BAAI/Emu3-VisionTokenizer; Wang et al. 2024) as plain PyTorch.
# A MoVQGAN-style autoencoder. The image is resized so both sides are multiples of 8 and its area lies between 512²
# and 1024² pixels, encoded 8× smaller into 4 numbers per position, and each position is replaced by the id of the
# nearest of 32,768 codebook vectors. The decoder turns the ids back into pixels; every normalisation in it is
# modulated by the looked-up codes. The tokenizer also handles video (4 frames → 1 latent frame), so an image goes
# in as a still clip of 4 frames. Emu3's language model reads the ids row by row as visual tokens, each row ended
# by a row-end token. Each "# %% [id]" section is one step of the Emu3 tab.
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


@dataclass
class Config:
    ch: int = 256
    ch_mult: tuple = (1, 2, 2, 4)  # widths 256, 512, 512, 1024 at 1, 1/2, 1/4 and 1/8 of the image
    res_blocks: int = 2
    attn_levels: tuple = (3,)  # attention only at the 1/8 level
    z_channels: int = 4
    embed_dim: int = 4  # width of one codebook vector
    codebook: int = 32768
    frames: int = 4  # frames per latent frame (temporal_downsample_factor)
    factor: int = 8  # spatial downsampling
    min_pixels: int = 512 * 512
    max_pixels: int = 1024 * 1024


def silu(x):
    return x * torch.sigmoid(x)


# %% [preprocess] Resize to multiples of 8, scale to [−1, 1]
def smart_resize(height, width, c: Config):
    """Round both sides to multiples of 8; if the area leaves [min_pixels, max_pixels], rescale it into range."""
    h, w = round(height / c.factor) * c.factor, round(width / c.factor) * c.factor
    if h * w > c.max_pixels:
        beta = math.sqrt(height * width / c.max_pixels)
        h, w = math.floor(height / beta / c.factor) * c.factor, math.floor(width / beta / c.factor) * c.factor
    elif h * w < c.min_pixels:
        beta = math.sqrt(c.min_pixels / (height * width))
        h, w = math.ceil(height * beta / c.factor) * c.factor, math.ceil(width * beta / c.factor) * c.factor
    return h, w


def preprocess(image: Image.Image, c: Config):
    """PIL image → (1, 3, h, w) in [−1, 1]: bicubic resize, ÷ 255, then (x − 0.5) / 0.5."""
    image = image.convert("RGB")
    h, w = smart_resize(image.height, image.width, c)
    pixels = np.asarray(image.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255
    return torch.from_numpy((pixels - 0.5) / 0.5).permute(2, 0, 1)[None]


# %% [encoder] Encoder: image → 4 numbers per 8 × 8 pixels
class CausalConv3d(nn.Module):
    """A 3D convolution padded only towards the past in time (2 frames), so no frame sees a later one."""

    def __init__(self, cin, cout, kernel=(3, 1, 1), stride=(1, 1, 1)):
        super().__init__()
        ph, pw = kernel[1] - stride[1], kernel[2] - stride[2]
        self.padding = (pw // 2 + pw % 2, pw // 2, ph // 2 + ph % 2, ph // 2, 2, 0)
        self.conv = nn.Conv3d(cin, cout, kernel, stride=stride)

    def forward(self, x):  # (b, c, t, h, w)
        return self.conv(F.pad(x, self.padding))


class TemporalResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm1, self.conv1 = nn.BatchNorm3d(ch), CausalConv3d(ch, ch, (3, 3, 3))
        self.norm2, self.conv2 = nn.BatchNorm3d(ch), CausalConv3d(ch, ch, (3, 3, 3))

    def forward(self, x):
        h = self.conv1(silu(self.norm1(x)))
        return x + self.conv2(silu(self.norm2(h)))


class TemporalDownsample(nn.Module):
    """Halves the frames: kernel 4 in time, stride 2."""

    def __init__(self, ch):
        super().__init__()
        self.conv = CausalConv3d(ch, ch, (4, 3, 3), (2, 1, 1))

    def forward(self, x):
        return self.conv(x)


class SpatialNorm(nn.Module):
    """The decoder's GroupNorm: scale and shift come from the looked-up codes, nearest-upsampled to the feature map,
    so every layer sees the quantized latent, not only the first (MoVQ)."""

    def __init__(self, ch, zq_ch):
        super().__init__()
        self.norm_layer = nn.GroupNorm(32, ch, eps=1e-6)
        self.conv_y, self.conv_b = nn.Conv2d(zq_ch, ch, 1), nn.Conv2d(zq_ch, ch, 1)

    def forward(self, x, zq):
        zq = F.interpolate(zq, size=x.shape[-2:], mode="nearest")
        return self.norm_layer(x) * self.conv_y(zq) + self.conv_b(zq)


def norm(ch, zq_ch):
    return nn.GroupNorm(32, ch, eps=1e-6) if zq_ch is None else SpatialNorm(ch, zq_ch)


class ResBlock(nn.Module):
    """Two 3 × 3 convolutions after GroupNorm (SpatialNorm in the decoder) and SiLU, added to the input."""

    def __init__(self, cin, cout, zq_ch=None):
        super().__init__()
        self.norm1, self.conv1 = norm(cin, zq_ch), nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2, self.conv2 = norm(cout, zq_ch), nn.Conv2d(cout, cout, 3, padding=1)
        self.nin_shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, zq=None):
        extra = () if zq is None else (zq,)
        h = self.conv1(silu(self.norm1(x, *extra)))
        h = self.conv2(silu(self.norm2(h, *extra)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    """One head over every position of the 1/8 map (94 × 125 = 11,750 positions for the beagle)."""

    def __init__(self, ch, zq_ch=None):
        super().__init__()
        self.norm = norm(ch, zq_ch)
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(ch, ch, 1) for _ in range(4))

    def forward(self, x, zq=None):
        b, c, h, w = x.shape
        y = self.norm(x) if zq is None else self.norm(x, zq)
        q, k, v = (f(y).reshape(b, c, h * w) for f in (self.q, self.k, self.v))
        weights = torch.softmax(q.transpose(1, 2) @ k / c**0.5, dim=2)  # (b, positions, positions)
        return x + self.proj_out((v @ weights.transpose(1, 2)).reshape(b, c, h, w))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))


def level(blocks, attn, extra=None):
    m = nn.Module()
    m.block, m.attn = nn.ModuleList(blocks), nn.ModuleList(attn)
    for name, module in (extra or {}).items():
        setattr(m, name, module)
    return m


class Encoder(nn.Module):
    """Per frame: four levels of ResNet blocks with a stride-2 convolution between them (attention at 1/8), then a
    3 × 3 convolution to 4 channels. Then across frames: two causal convolutions halve 4 frames to 1, and two
    temporal ResNet blocks follow."""

    def __init__(self, c: Config):
        super().__init__()
        widths = [c.ch] + [c.ch * m for m in c.ch_mult]
        last = len(c.ch_mult) - 1
        self.conv_in = nn.Conv2d(3, c.ch, 3, padding=1)
        self.down = nn.ModuleList(
            level(
                [ResBlock(widths[i] if j == 0 else widths[i + 1], widths[i + 1]) for j in range(c.res_blocks)],
                [AttnBlock(widths[i + 1]) for _ in range(c.res_blocks if i in c.attn_levels else 0)],
                {"downsample": Downsample(widths[i + 1])} if i < last else None,
            )
            for i in range(len(c.ch_mult))
        )
        top = widths[-1]
        self.mid = level([], [], {"block_1": ResBlock(top, top), "attn_1": AttnBlock(top), "block_2": ResBlock(top, top)})
        self.norm_out = nn.GroupNorm(32, top, eps=1e-6)
        self.conv_out = nn.Conv2d(top, c.z_channels, 3, padding=1)
        self.time_conv = nn.ModuleList(TemporalDownsample(c.z_channels) for _ in range(int(math.log2(c.frames))))
        self.time_res_stack = nn.Sequential(*(TemporalResBlock(c.z_channels) for _ in range(c.res_blocks)))

    def forward(self, x):  # (b, t, 3, H, W) → (b, t / 4, 4, H / 8, W / 8)
        b, t = x.shape[:2]
        h = self.conv_in(x.flatten(0, 1))
        for down in self.down:
            for j, block in enumerate(down.block):
                h = block(h)
                if len(down.attn):
                    h = down.attn[j](h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        h = self.conv_out(silu(self.norm_out(h)))
        h = h.unflatten(0, (b, t)).transpose(1, 2)  # (b, 4, t, h, w)
        for conv in self.time_conv:
            h = silu(conv(h))
        return self.time_res_stack(h).transpose(1, 2)


# %% [quantize] Nearest of 32,768 codes → ids
class Quantizer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embedding = nn.Embedding(c.codebook, c.embed_dim)

    def distances(self, z):
        """z: (b, t, 4, h, w) → squared distance from every position to every code, (b·t·h·w, 32768), written
        ‖x‖² + ‖e‖² − 2 x·e as Emu3 computes it."""
        x = z.permute(0, 1, 3, 4, 2).reshape(-1, z.shape[2])
        e = self.embedding.weight
        return (x**2).sum(1, keepdim=True) + (e**2).sum(1) - 2 * x @ e.T

    def forward(self, z):  # → ids (b, t, h, w)
        b, t, _, h, w = z.shape
        return self.distances(z).argmin(1).view(b, t, h, w)


# %% [decoder] Decoder: ids → image
class TemporalUpsample(nn.Module):
    """Doubles the frames (each repeated), then a causal 3 × 3 × 3 convolution."""

    def __init__(self, ch):
        super().__init__()
        self.conv = CausalConv3d(ch, ch, (3, 3, 3))

    def forward(self, x):  # (b, c, t, h, w)
        return self.conv(x.repeat_interleave(2, dim=2))


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Decoder(nn.Module):
    """The encoder mirrored: temporal blocks and two upsamplings in time (1 latent frame → 4 frames), then per
    frame a 3 × 3 convolution to 1024 channels, attention, and four levels of three ResNet blocks with nearest ×2
    upsampling between them. The looked-up codes go through the same temporal layers and condition every
    SpatialNorm."""

    def __init__(self, c: Config):
        super().__init__()
        zq, top = c.embed_dim, c.ch * c.ch_mult[-1]
        self.time_res_stack = nn.Sequential(*(TemporalResBlock(c.z_channels) for _ in range(c.res_blocks)))
        self.time_conv = nn.ModuleList(TemporalUpsample(c.z_channels) for _ in range(int(math.log2(c.frames))))
        self.conv_in = nn.Conv2d(c.z_channels, top, 3, padding=1)
        self.mid = level(
            [], [], {"block_1": ResBlock(top, top, zq), "attn_1": AttnBlock(top, zq), "block_2": ResBlock(top, top, zq)}
        )
        ups, width = [], top
        for i in reversed(range(len(c.ch_mult))):
            out = c.ch * c.ch_mult[i]
            blocks = [ResBlock(width if j == 0 else out, out, zq) for j in range(c.res_blocks + 1)]
            attn = [AttnBlock(out, zq) for _ in range(c.res_blocks + 1 if i in c.attn_levels else 0)]
            ups.insert(0, level(blocks, attn, {"upsample": Upsample(out)} if i else None))
            width = out
        self.up = nn.ModuleList(ups)
        self.norm_out = SpatialNorm(width, zq)
        self.conv_out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, z, zq):  # both (b, t, 4, h, w): post_quant_conv(codes) and the codes
        both = self.time_res_stack(torch.cat([z, zq]).transpose(1, 2))
        for conv in self.time_conv:
            both = silu(conv(both))
        h, zq = (part.flatten(0, 1) for part in both.transpose(1, 2).chunk(2))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(h), zq), zq), zq)
        for up in reversed(self.up):
            for j, block in enumerate(up.block):
                h = block(h, zq)
                if len(up.attn):
                    h = up.attn[j](h, zq)
            if hasattr(up, "upsample"):
                h = up.upsample(h)
        h = self.conv_out(silu(self.norm_out(h, zq)))
        return h.unflatten(0, (z.shape[0], -1))  # (b, 4 t, 3, H, W)


# %% [sequence] Ids → what the language model reads
VISUAL_0 = 151854  # <|visual token 000000|>; the 32,768 visual tokens follow Emu3's text vocabulary
BOI, IMG, EOL, EOF, EOI = 151852, 151851, 151846, 151847, 151853  # image start, image token, row end, frame end, image end


def image_tokens(ids, encode_text):
    """Emu3's processor writes an image as text: <|image start|>, "H*W" in ordinary digits, <|image token|>, the
    rows of visual tokens each followed by a row-end token, then end of frame and <|image end|>. \`encode_text\`
    is Emu3's tiktoken BPE."""
    h, w = ids.shape
    rows = [[VISUAL_0 + int(i) for i in row] + [EOL] for row in ids]
    return [BOI, *encode_text(f"{h}*{w}"), IMG, *(t for row in rows for t in row), EOF, EOI]


# %% [model] Emu3's vision tokenizer
class Emu3VisionTokenizer(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.encoder, self.decoder, self.quantize = Encoder(c), Decoder(c), Quantizer(c)
        self.quant_conv = CausalConv3d(c.z_channels, c.embed_dim)
        self.post_quant_conv = CausalConv3d(c.embed_dim, c.z_channels)

    def latent(self, image):
        """(b, 3, H, W) → (b, 1, 4, H / 8, W / 8): the vectors the codebook is searched with."""
        clip = image[:, None].repeat(1, self.c.frames, 1, 1, 1)  # an image is a still clip of 4 frames
        return self.quant_conv(self.encoder(clip).transpose(1, 2)).transpose(1, 2)

    def encode(self, image):  # → ids (b, H / 8, W / 8)
        return self.quantize(self.latent(image))[:, 0]

    def decode(self, ids):  # (b, h, w) → (b, 3, 8 h, 8 w)
        codes = self.quantize.embedding(ids[:, None]).permute(0, 1, 4, 2, 3)  # (b, 1, 4, h, w)
        z = self.post_quant_conv(codes.transpose(1, 2)).transpose(1, 2)
        return self.decoder(z, codes)[:, 0]  # the first of the 4 frames the decoder makes
`;export{n as default};
