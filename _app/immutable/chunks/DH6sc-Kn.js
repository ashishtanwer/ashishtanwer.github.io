const e=`# Cosmos-Predict1-4B (NVIDIA 2025), the autoregressive base world model, as plain PyTorch.
# A causal video tokenizer (Cosmos-Tokenize1-DV8x16x16) turns 33 frames of 640 × 1024 into 5 × 40 × 64 discrete
# tokens: a 3D Haar wavelet, a factorized causal conv encoder and finite scalar quantization (6 numbers rounded to
# 8·8·8·5·5·5 levels, one id out of 64000). A 4B Llama-style transformer with 3D RoPE reads the tokens of the first
# frames and samples the rest one token at a time; the tokenizer's decoder turns them back into video.
# Each "# %% [id]" section is one step of the Cosmos AR tab's diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # tokenizer
    channels: int = 128
    channels_mult: tuple = (2, 4, 4)  # three levels: 128 → 256 → 512 → 512 channels
    res_blocks: int = 2  # per level in the encoder; one more per level in the decoder
    patch: int = 4  # the Haar wavelet's 4 × 4 × 4 patches
    spatial_downs: int = 2  # 16 = 4 (wavelet) × 2 × 2
    temporal_downs: int = 1  # 8 = 4 (wavelet) × 2
    z_channels: int = 16
    levels: tuple = (8, 8, 8, 5, 5, 5)  # FSQ: 6 numbers per token, 64000 ids
    # transformer
    vocab: int = 64000
    width: int = 4096
    layers: int = 16
    heads: int = 32
    kv_heads: int = 8
    ffn: int = 14336
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    latent: tuple = (5, 40, 64)  # latent frames × rows × columns
    top_p: float = 0.8
    temperature: float = 1.0

    @property
    def head_dim(self):
        return self.width // self.heads


# %% [patch] 3D Haar wavelet: 33 × 640 × 1024 × 3 → 9 × 160 × 256 × 192
def haar(x, inverse=False):
    """One level of the 3D Haar transform along time, height and width: each 2 × 2 × 2 cube becomes 8 bands
    (low/high in t, then h, then w), scaled by 1/(2√2). The inverse puts the cubes back."""
    s = 2**0.5
    if not inverse:  # (B, C, T, H, W) → (B, 8C, T/2, H/2, W/2)
        bands = [x]
        for dim in (2, 3, 4):
            pairs = [band.unflatten(dim, (-1, 2)).unbind(dim + 1) for band in bands]
            bands = [half for a, b in pairs for half in ((a + b) / s, (a - b) / s)]
        return torch.cat(bands, 1) / (2 * s)
    bands = list((x * (2 * s)).chunk(8, 1))  # (B, 8C, T, H, W) → (B, C, 2T, 2H, 2W)
    for dim in (4, 3, 2):
        bands = [torch.stack(((lo + hi) / s, (lo - hi) / s), dim + 1).flatten(dim, dim + 1) for lo, hi in zip(bands[::2], bands[1::2])]
    return bands[0]


def patchify(video, c):
    """The first frame is repeated 4 times, so it gets a latent frame of its own: 33 frames → 36 → 9."""
    video = torch.cat([video[:, :, :1].repeat_interleave(c.patch, 2), video[:, :, 1:]], 2)
    for _ in range(c.patch.bit_length() - 1):
        video = haar(video)
    return video


def unpatchify(x, c):
    for _ in range(c.patch.bit_length() - 1):
        x = haar(x, inverse=True)
    return x[:, :, c.patch - 1 :]  # the first frame's copies collapse to one


# %% [encoder] Causal encoder: 9 × 160 × 256 → 5 × 40 × 64 × 16
class CausalConv(nn.Module):
    """A 3D convolution that only looks back in time: the first frame is repeated in front, never a future one."""

    def __init__(self, cin, cout, kernel=(3, 3, 3), stride=1, time_stride=1, padding=None):
        super().__init__()
        kt, kh, _ = kernel
        self.time_pad = kt - time_stride
        self.pad = kh // 2 if padding is None else padding
        self.conv3d = nn.Conv3d(cin, cout, kernel, stride=(time_stride, stride, stride))

    def forward(self, x):
        x = torch.cat([x[:, :, :1].repeat(1, 1, self.time_pad, 1, 1), x], 2)
        return self.conv3d(F.pad(x, (self.pad,) * 4))


def factorized(cin, cout):
    """3 × 3 in space, then 3 in time."""
    return nn.Sequential(CausalConv(cin, cout, (1, 3, 3)), CausalConv(cout, cout, (3, 1, 1)))


class FrameNorm(nn.Module):
    """GroupNorm with one group, per frame (so it stays causal)."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(1, ch, eps=1e-6)

    def forward(self, x):
        b = x.shape[0]
        return self.norm(x.transpose(1, 2).flatten(0, 1)).unflatten(0, (b, -1)).transpose(1, 2)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1, self.conv1 = FrameNorm(cin), factorized(cin, cout)
        self.norm2, self.conv2 = FrameNorm(cout), factorized(cout, cout)
        self.nin_shortcut = CausalConv(cin, cout, (1, 1, 1)) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return self.nin_shortcut(x) + self.conv2(F.silu(self.norm2(h)))


class Attention3d(nn.Module):
    """One head over the pixels of each frame (space), or over each pixel's frames with a causal mask (time)."""

    def __init__(self, ch, over):
        super().__init__()
        self.over = over
        self.norm = FrameNorm(ch)
        self.q, self.k, self.v, self.proj_out = (CausalConv(ch, ch, (1, 1, 1)) for _ in range(4))

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = self.norm(x)
        if self.over == "space":  # (B·T, H·W, C)
            q, k, v = (f(y).permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c) for f in (self.q, self.k, self.v))
            out = F.scaled_dot_product_attention(q, k, v).reshape(b, t, h, w, c)
        else:  # (B·H·W, T, C)
            q, k, v = (f(y).permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c) for f in (self.q, self.k, self.v))
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True).reshape(b, h, w, t, c).permute(0, 3, 1, 2, 4)
        return x + self.proj_out(out.permute(0, 4, 1, 2, 3))


class Downsample(nn.Module):
    """Stride 2 in space and/or time, each a strided conv plus an average pool."""

    def __init__(self, ch, spatial, temporal):
        super().__init__()
        self.spatial, self.temporal = spatial, temporal
        self.conv1 = CausalConv(ch, ch, (1, 3, 3), stride=2, padding=0) if spatial else None
        self.conv2 = CausalConv(ch, ch, (3, 1, 1), time_stride=2, padding=0) if temporal else None
        self.conv3 = CausalConv(ch, ch, (1, 1, 1))

    def forward(self, x):
        if self.spatial:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.conv1(x) + F.avg_pool3d(x, (1, 2, 2), (1, 2, 2))
        if self.temporal:
            x = torch.cat([x[:, :, :1], x], 2)
            x = self.conv2(x) + F.avg_pool3d(x, (2, 1, 1), (2, 1, 1))
        return self.conv3(x)


class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        ch = [c.channels * m for m in (1, *c.channels_mult)]
        self.conv_in = factorized(3 * c.patch**3, ch[0])
        self.down = nn.ModuleList()
        for i in range(len(c.channels_mult)):
            level = nn.Module()
            level.block = nn.ModuleList(ResBlock(ch[i] if j == 0 else ch[i + 1], ch[i + 1]) for j in range(c.res_blocks))
            if i < len(c.channels_mult) - 1:
                level.downsample = Downsample(ch[i + 1], i < c.spatial_downs, i < c.temporal_downs)
            self.down.append(level)
        self.mid = nn.Sequential(ResBlock(ch[-1], ch[-1]), Attention3d(ch[-1], "space"), Attention3d(ch[-1], "time"), ResBlock(ch[-1], ch[-1]))
        self.norm_out, self.conv_out = FrameNorm(ch[-1]), factorized(ch[-1], c.z_channels)
        self.quant_conv = CausalConv(c.z_channels, len(c.levels), (1, 1, 1))

    def forward(self, x):  # x: patchified video
        h = self.conv_in(x)
        for level in self.down:
            for block in level.block:
                h = block(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        h = self.conv_out(F.silu(self.norm_out(self.mid(h))))
        return self.quant_conv(h)  # (B, 6, T, H, W): the numbers FSQ rounds


# %% [quantize] FSQ: 6 numbers → 6 digits → one id of 64000
def fsq(z, c):
    """z: (..., 6). Squash each number into its level's range, round it, read the digits as one mixed-radix id."""
    levels = torch.tensor(c.levels)
    half = (levels - 1) * (1 + 1e-3) / 2
    offset = (levels % 2 == 0) * 0.5  # even levels are shifted so the rounding lands on whole steps
    shift = torch.atanh(offset / half)
    bounded = torch.tanh(z + shift) * half - offset
    digits = bounded.round() + levels // 2  # 0 … level − 1
    basis = torch.cumprod(torch.tensor([1, *c.levels[:-1]]), 0)
    ids = (digits * basis).sum(-1).long()
    return ids, bounded


def ids_to_codes(ids, c):
    """The decoder's input: each id's 6 digits, rescaled to [-1, 1]."""
    levels = torch.tensor(c.levels)
    basis = torch.cumprod(torch.tensor([1, *c.levels[:-1]]), 0)
    digits = (ids[..., None] // basis) % levels
    return (digits - levels // 2) / (levels // 2)


# %% [decoder] Causal decoder: 5 × 40 × 64 tokens → 33 frames
class Upsample(nn.Module):
    """Repeat ×2 in time (all but the first frame) and/or space, each followed by a residual conv."""

    def __init__(self, ch, spatial, temporal):
        super().__init__()
        self.spatial, self.temporal = spatial, temporal
        self.conv1 = CausalConv(ch, ch, (3, 1, 1), padding=0) if temporal else None
        self.conv2 = CausalConv(ch, ch, (1, 3, 3)) if spatial else None
        self.conv3 = CausalConv(ch, ch, (1, 1, 1))

    def forward(self, x):
        if self.temporal:
            if x.shape[2] > 1:
                x = x.repeat_interleave(2, 2)[:, :, 1:]
            x = self.conv1(x) + x
        if self.spatial:
            x = x.repeat_interleave(2, 3).repeat_interleave(2, 4)
            x = self.conv2(x) + x
        return self.conv3(x)


class Decoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        ch = [c.channels * m for m in c.channels_mult]
        n = len(ch)
        self.post_quant_conv = CausalConv(len(c.levels), c.z_channels, (1, 1, 1))
        self.conv_in = factorized(c.z_channels, ch[-1])
        self.mid = nn.Sequential(ResBlock(ch[-1], ch[-1]), Attention3d(ch[-1], "space"), Attention3d(ch[-1], "time"), ResBlock(ch[-1], ch[-1]))
        self.up = nn.ModuleList()
        block_in = ch[-1]
        for i in reversed(range(n)):  # the encoder's levels mirrored: 512 → 512 → 256
            level = nn.Module()
            level.block = nn.ModuleList()
            for _ in range(c.res_blocks + 1):
                level.block.append(ResBlock(block_in, ch[i]))
                block_in = ch[i]
            if i > 0:
                temporal = 0 < n - 1 - i <= c.temporal_downs
                spatial = temporal or (n - 1 - i < c.spatial_downs and c.spatial_downs > c.temporal_downs)
                level.upsample = Upsample(block_in, spatial, temporal)
            self.up.insert(0, level)
        self.norm_out, self.conv_out = FrameNorm(block_in), factorized(block_in, 3 * c.patch**3)

    def forward(self, codes):  # (B, 6, T, H, W)
        h = self.mid(self.conv_in(self.post_quant_conv(codes)))
        for level in reversed(self.up):
            for block in level.block:
                h = block(h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))  # wavelet bands; unpatchify gives frames in [-1, 1]


class Tokenizer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c, self.encoder, self.decoder = c, Encoder(c), Decoder(c)

    def encode(self, video):  # (B, 3, 33, H, W) in [-1, 1] → ids (B, 5, H/16, W/16)
        return fsq(self.encoder(patchify(video, self.c)).movedim(1, -1), self.c)[0]

    def decode(self, ids):
        return unpatchify(self.decoder(ids_to_codes(ids, self.c).movedim(-1, 1).float()), self.c)


# %% [embedding] Token embedding: one 4096-wide vector per id
def embed(table: nn.Embedding, ids):
    return table(ids)  # (B, S, 4096); there is no position embedding: positions enter through RoPE


# %% [rope] 3D RoPE: every token's (t, row, column)
def rope_angles(c: Config):
    """(T·H·W, head_dim) rotation angles. A head's 128 dimensions are split 44 / 42 / 42 between time, rows and
    columns; each pair rotates at its own frequency times that coordinate."""
    d_hw = c.head_dim // 6 * 2
    d_t = c.head_dim - 2 * d_hw

    def inv_freq(d):
        return 1.0 / (c.rope_theta ** (torch.arange(0, d, 2)[: d // 2].float() / d))

    t, h, w = torch.meshgrid(*(torch.arange(n, dtype=torch.float) for n in c.latent), indexing="ij")
    half = torch.cat([t.reshape(-1, 1) * inv_freq(d_t), h.reshape(-1, 1) * inv_freq(d_hw), w.reshape(-1, 1) * inv_freq(d_hw)], -1)
    return torch.cat([half, half], -1)


def rotate(x, angles):
    """x: (B, S, heads, head_dim); the second half of each head pairs with the first (rotate-half)."""
    x1, x2 = x.chunk(2, -1)
    cos, sin = angles.cos()[None, :, None], angles.sin()[None, :, None]
    return x * cos + torch.cat([-x2, x1], -1) * sin


# %% [attention] Grouped-query attention with QK-norm and a KV cache
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class Attention(nn.Module):
    """32 query heads share 8 key/value heads; queries and keys are RMS-normalized per head before RoPE."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wq = nn.Linear(c.width, c.heads * c.head_dim, bias=False)
        self.wk = nn.Linear(c.width, c.kv_heads * c.head_dim, bias=False)
        self.wv = nn.Linear(c.width, c.kv_heads * c.head_dim, bias=False)
        self.wo = nn.Linear(c.heads * c.head_dim, c.width, bias=False)
        self.q_norm, self.k_norm = RMSNorm(c.head_dim, c.norm_eps), RMSNorm(c.head_dim, c.norm_eps)

    def forward(self, x, angles, cache):
        """x: (B, S, 4096) for the next S positions; angles: their RoPE angles; cache: this layer's keys and values."""
        b, s, _ = x.shape
        q = self.q_norm(self.wq(x).view(b, s, self.c.heads, -1))
        k = self.k_norm(self.wk(x).view(b, s, self.c.kv_heads, -1))
        v = self.wv(x).view(b, s, self.c.kv_heads, -1)
        q, k = rotate(q, angles), rotate(k, angles)
        cache["k"] = k if "k" not in cache else torch.cat([cache["k"], k], 1)
        cache["v"] = v if "v" not in cache else torch.cat([cache["v"], v], 1)
        group = self.c.heads // self.c.kv_heads
        keys, values = (cache[n].repeat_interleave(group, 2).transpose(1, 2) for n in ("k", "v"))
        seen = keys.shape[2]
        mask = torch.ones(s, seen, dtype=torch.bool).tril(seen - s)  # position i sees every key up to i
        out = F.scaled_dot_product_attention(q.transpose(1, 2), keys, values, attn_mask=mask)
        return self.wo(out.transpose(1, 2).reshape(b, s, -1))


# %% [mlp] SwiGLU feed-forward: 4096 → 14336 → 4096
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.w1 = nn.Linear(c.width, c.ffn, bias=False)
        self.w3 = nn.Linear(c.width, c.ffn, bias=False)
        self.w2 = nn.Linear(c.ffn, c.width, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# %% [block] Transformer block (pre-norm)
class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.attention_norm, self.attention = RMSNorm(c.width, c.norm_eps), Attention(c)
        self.ffn_norm, self.feed_forward = RMSNorm(c.width, c.norm_eps), MLP(c)

    def forward(self, x, angles, cache):
        h = x + self.attention(self.attention_norm(x), angles, cache)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok_embeddings = nn.Embedding(c.vocab, c.width)
        self.layers = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.width, c.norm_eps)
        self.output = nn.Linear(c.width, c.vocab, bias=False)
        self.register_buffer("angles", rope_angles(c), persistent=False)

    def forward(self, ids, start, caches):
        """ids: (B, S) at positions start … start + S − 1 → logits (B, S, 64000) for the token after each."""
        x = embed(self.tok_embeddings, ids)
        angles = self.angles[start : start + ids.shape[1]]
        for layer, cache in zip(self.layers, caches):
            x = layer(x, angles, cache)
        return self.output(self.norm(x))


# %% [output] Top-p sampling: keep the most likely ids up to 80% of the mass
def sample_top_p(logits, c: Config):
    """logits: (B, 64000) for one position. Sort, cut the tail past top_p, renormalize, draw one id (Gumbel-style:
    argmax of p / Exp(1) noise is a sample from p)."""
    probs = torch.softmax(logits / c.temperature, -1)
    p_sort, p_idx = probs.sort(-1, descending=True)
    p_sort[p_sort.cumsum(-1) - p_sort > c.top_p] = 0.0
    p_sort = p_sort / p_sort.sum(-1, keepdim=True)
    pick = torch.argmax(p_sort / torch.empty_like(p_sort).exponential_(1), -1, keepdim=True)
    return p_idx.gather(-1, pick)


# %% [model] Video continuation: prompt tokens in, the remaining latent frames out
@torch.no_grad()
def generate(transformer: Transformer, prompt, new_tokens, c: Config):
    """prompt: (B, P) ids, e.g. the first 2 latent frames (5120 tokens). Reads them in one pass (prefill), then
    samples new_tokens more, each from one forward pass over the token before it and the cached keys/values."""
    caches = [{} for _ in transformer.layers]
    tokens = [sample_top_p(transformer(prompt, 0, caches)[:, -1], c)]
    for i in range(new_tokens - 1):
        logits = transformer(tokens[-1], prompt.shape[1] + i, caches)
        tokens.append(sample_top_p(logits[:, -1], c))
    return torch.cat([prompt, *tokens], 1)  # (B, 5 · 40 · 64) for the tokenizer's decoder
`;export{e as default};
