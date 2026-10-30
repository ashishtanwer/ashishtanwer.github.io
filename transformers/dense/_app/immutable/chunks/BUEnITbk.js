const e=`# KV-cache quantization (KIVI) on Qwen2.5-0.5B-Instruct, the model in the KV-quant tab, as plain PyTorch.
# The cache stores keys and values as 8-, 4- or 2-bit integers with a float16 scale and zero point per
# group: keys per channel over groups of 16 tokens, values per token. Attention reads them dequantized.
# The model is the KV-cache tab's; only the cache changes. Each "# %% [id]" section is one block.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Qwen2.5-0.5B-Instruct
    vocab_size: int = 151936
    d_model: int = 896
    n_layers: int = 24
    n_heads: int = 14  # 64 dimensions per head
    n_kv_heads: int = 2  # seven query heads share each key/value head
    d_mlp: int = 4864
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    bytes_per_element: int = 2  # bfloat16, for the rows kept in full precision

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


# %% [quantize] Min–max quantization
def quantize(x, bits, dim):
    """Asymmetric: the group's minimum is the zero point and (max − min) / (2^bits − 1) the step, both
    stored as float16; each value becomes the nearest integer 0 … 2^bits − 1. \`dim\` is the group."""
    levels = 2**bits - 1
    zero = x.amin(dim, keepdim=True).half()
    scale = ((x.amax(dim, keepdim=True) - zero.float()) / levels).half()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = ((x - zero.float()) / scale.float()).round().clamp(0, levels)
    return codes, scale, zero


def dequantize(codes, scale, zero):
    return codes * scale.float() + zero.float()


# %% [kivi] KIVI cache
class KiviCache:
    """Per layer: finished groups as integers, plus the newest rows in full precision.

    Keys have a few channels far larger than the rest in every token, so they are grouped per channel
    (one scale per channel over \`group\` consecutive tokens); values have no such channels and are
    grouped per token (one scale over the head's 64 channels). Rows wait in full precision until
    \`group\` of them have arrived (KIVI's residual), then the group is quantized. keys="token" groups
    the keys per token too, for comparison."""

    def __init__(self, c: Config, bits=4, group=16, keys="channel"):
        self.c, self.bits, self.group, self.keys = c, bits, group, keys
        self.groups = [[] for _ in range(c.n_layers)]  # [(k codes, k scale, k zero, v codes, v scale, v zero)]
        self.residual = [None] * c.n_layers  # (k, v), each (n_kv_heads, rows, head_dim)

    def quantize_group(self, k, v):  # (n_kv_heads, group, head_dim)
        key_dim = 1 if self.keys == "channel" else 2  # over the group's tokens, or over a token's channels
        return (*quantize(k, self.bits, key_dim), *quantize(v, self.bits, 2))

    def read(self, layer):
        """What attention reads: every finished group dequantized, then the full-precision rows."""
        ks = [dequantize(*g[:3]) for g in self.groups[layer]]
        vs = [dequantize(*g[3:]) for g in self.groups[layer]]
        if self.residual[layer] is not None:
            ks.append(self.residual[layer][0])
            vs.append(self.residual[layer][1])
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)

    def append(self, layer, k, v):
        """The new rows are read in full precision by this step; afterwards every full group of the
        residual is quantized. The prefill therefore attends in full precision, as KIVI's does."""
        if self.residual[layer] is not None:
            k, v = torch.cat((self.residual[layer][0], k), dim=1), torch.cat((self.residual[layer][1], v), dim=1)
        self.residual[layer] = (k, v)
        keys, values = self.read(layer)
        while self.residual[layer][0].shape[1] >= self.group:
            k, v = self.residual[layer]
            self.groups[layer].append(self.quantize_group(k[:, : self.group], v[:, : self.group]))
            self.residual[layer] = (k[:, self.group :], v[:, self.group :])
        return keys, values

    def __len__(self):
        return len(self.groups[0]) * self.group + (0 if self.residual[0] is None else self.residual[0][0].shape[1])


# %% [kv-bytes] Bytes stored
def kivi_bytes(c: Config, rows, bits, group=16, keys="channel"):
    """Integers, float16 scales and zero points, and the newest rows in bfloat16, for all layers."""
    n = rows // group * group  # rows in finished groups
    integers = 2 * c.n_layers * c.n_kv_heads * n * c.head_dim * bits // 8
    key_scales = c.n_layers * c.n_kv_heads * ((n // group) * c.head_dim if keys == "channel" else n) * 2 * 2
    value_scales = c.n_layers * c.n_kv_heads * n * 2 * 2
    residual = 2 * c.n_layers * c.n_kv_heads * (rows - n) * c.head_dim * c.bytes_per_element
    return integers + key_scales + value_scales + residual


# %% [attention] Attention over the dequantized cache
def rope(x, positions, theta):
    d = x.shape[-1]
    frequencies = 1.0 / theta ** (torch.arange(0, d, 2, device=x.device).float() / d)
    angles = positions[:, None].float() * frequencies
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def forward(self, x, positions, cache, layer):
        c, t = self.c, x.shape[0]
        q = self.q(x).view(t, c.n_heads, c.head_dim).transpose(0, 1)
        k = self.k(x).view(t, c.n_kv_heads, c.head_dim).transpose(0, 1)
        v = self.v(x).view(t, c.n_kv_heads, c.head_dim).transpose(0, 1)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        k, v = cache.append(layer, k, v)  # dequantized groups + full-precision rows
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=0), v.repeat_interleave(group, dim=0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(c.head_dim)
        visible = torch.arange(k.shape[1], device=x.device) <= positions[:, None]
        weights = scores.masked_fill(~visible, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(0, 1).reshape(t, -1)), weights


# %% [block] Transformer block
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, positions, cache, layer):
        x = x + self.attention(self.norm1(x), positions, cache, layer)[0]
        return x + self.mlp(self.norm2(x))


# %% [model] Qwen2.5 reading the quantized cache
class Qwen2(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, ids, positions, cache):
        x = self.embedding(ids)
        for layer, block in enumerate(self.blocks):
            x = block(x, positions, cache, layer)
        return self.lm_head(self.norm(x))


# %% [generate] Greedy decoding with the quantized cache
@torch.no_grad()
def generate(model, prompt, cache, max_new_tokens=24, eos=151645):
    """Greedy decoding exactly as with a plain cache; only what attention reads has changed."""
    logits = model(prompt, torch.arange(len(prompt)), cache)[-1]
    tokens = []
    while len(tokens) < max_new_tokens:
        token = logits.argmax()
        tokens.append(int(token))
        if token == eos or len(tokens) == max_new_tokens:
            break
        logits = model(token[None], torch.tensor([len(cache)]), cache)[-1]
    return tokens


# tokens = generate(Qwen2(), prompt_ids, KiviCache(Config(), bits=4))  # keys per channel, values per token
`;export{e as default};
