const e=`# LayerSkip on Llama 3.2 1B, the Efficiency Explainer's LayerSkip tab, written as plain PyTorch blocks.
# facebook/layerskip-llama3.2-1B was trained so that the final RMSNorm and LM head read the hidden state after any
# layer (early exit). Self-speculation drafts with the first E layers and verifies with the rest, reusing the
# draft's keys, values and layer-E states; no second model.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # facebook/layerskip-llama3.2-1B (Llama 3.2 1B's shape)
    vocab_size: int = 128256
    d_model: int = 2048
    n_layers: int = 16
    n_heads: int = 32  # 64 dimensions per head
    n_kv_heads: int = 8  # grouped-query attention: four query heads share each key/value head
    d_mlp: int = 8192
    rope_theta: float = 500000.0
    rope_factor: float = 32.0  # Llama 3's RoPE scaling for long contexts
    rope_low: float = 1.0
    rope_high: float = 4.0
    rope_context: int = 8192
    eps: float = 1e-5
    exit_layer: int = 3  # E: draft with layers 1 … 3 (the LayerSkip README's setting for this model)
    speculations: int = 3  # drafts per step
    eos: int = 128001  # <|end_of_text|>


# %% [block] Llama block
def rope_frequencies(c: Config, head_dim: int):
    """RoPE's per-pair frequencies with Llama 3's scaling: slow pairs divided by \`rope_factor\`, fast ones kept,
    the band between blended smoothly."""
    frequencies = 1.0 / c.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    wavelength = 2 * math.pi / frequencies
    slow, fast = c.rope_context / c.rope_low, c.rope_context / c.rope_high
    scaled = torch.where(wavelength > slow, frequencies / c.rope_factor, frequencies)
    blend = (c.rope_context / wavelength - c.rope_low) / (c.rope_high - c.rope_low)
    smoothed = (1 - blend) * scaled / c.rope_factor + blend * scaled
    middle = (wavelength >= fast) & (wavelength <= slow)
    return torch.where(middle, smoothed, scaled)


def rope(x, positions, frequencies):
    """Rotate each pair (i, i + d/2) of every head vector by position × frequency i. positions: (seq,)."""
    angles = positions.float()[:, None] * frequencies.to(x.device)  # (seq, d/2)
    angles = torch.cat((angles, angles), dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return x * angles.cos().to(x.dtype) + torch.cat((-x2, x1), dim=-1) * angles.sin().to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return self.weight * (x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype)


class Attention(nn.Module):
    """Grouped-query attention over the cached keys and values plus the new tokens, under a boolean mask."""

    def __init__(self, c: Config):
        super().__init__()
        self.c, self.head_dim = c, c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * self.head_dim, c.d_model, bias=False)
        self.register_buffer("frequencies", rope_frequencies(c, self.head_dim), persistent=False)

    def forward(self, x, positions, mask, cache=None):
        """x: (seq, d_model). mask: (seq, cached + seq). Returns the output and the new (k, v)."""
        c, t = self.c, x.shape[0]
        q = self.q(x).view(1, t, c.n_heads, -1).transpose(1, 2)  # (1, heads, seq, head_dim)
        k = self.k(x).view(1, t, c.n_kv_heads, -1).transpose(1, 2)
        v = self.v(x).view(1, t, c.n_kv_heads, -1).transpose(1, 2)
        q, k = rope(q, positions, self.frequencies), rope(k, positions, self.frequencies)
        keys = k if cache is None else torch.cat((cache[0], k), dim=2)
        values = v if cache is None else torch.cat((cache[1], v), dim=2)
        group = c.n_heads // c.n_kv_heads
        o = F.scaled_dot_product_attention(
            q, keys.repeat_interleave(group, 1), values.repeat_interleave(group, 1), attn_mask=mask
        )
        return self.out(o.transpose(1, 2).reshape(t, -1)), (k, v)


class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 2048 → 8192 → 2048."""

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

    def forward(self, x, positions, mask, cache=None):
        a, new = self.attention(self.norm1(x), positions, mask, cache)
        x = x + a
        return x + self.mlp(self.norm2(x)), new


class LayerSkipLlama(nn.Module):
    """The usual Llama, run one range of layers at a time; cache[l] holds layer l's keys and values."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # tied to the embedding in the checkpoint

    def layers(self, x, positions, cache, start, end):
        """Layers start … end − 1 on x (seq, d_model), causal over the cache and the new tokens."""
        n = x.shape[0]
        cached = 0 if cache[start] is None else cache[start][0].shape[2]
        mask = torch.ones(n, cached + n, dtype=torch.bool, device=x.device).tril(cached)
        for l in range(start, end):
            x, kv = self.blocks[l](x, positions, mask, cache[l])
            if cache[l] is not None:
                kv = tuple(torch.cat((old, new), dim=2) for old, new in zip(cache[l], kv))
            cache[l] = kv
        return x


def crop(cache, length):
    for l, kv in enumerate(cache):
        cache[l] = (kv[0][:, :, :length], kv[1][:, :, :length])


# %% [early-exit] Early exit after any layer
def early_exit(model, x):
    """The model's own final RMSNorm and LM head, applied after any layer. LayerSkip's early-exit loss trained every
    layer's output to be read this way; deeper layers predict better."""
    return model.lm_head(model.norm(x))


@torch.no_grad()
def every_exit(model, ids):
    """Next-token logits after each of the n_layers layers, for every position: (n_layers, seq, vocab)."""
    cache = [None] * model.c.n_layers
    x, positions, out = model.embedding(ids), torch.arange(len(ids)), []
    for l in range(model.c.n_layers):
        x = model.layers(x, positions, cache, l, l + 1)
        out.append(early_exit(model, x))
    return torch.stack(out)


# %% [draft] Draft with the first E layers
@torch.no_grad()
def draft(model, token, position, cache, c: Config):
    """Up to \`speculations\` tokens, one at a time through layers 1 … E and the early-exit head. Keeps each input's
    state after layer E (the exit query cache); layers 1 … E's keys and values go into the shared cache."""
    drafts, states = [], []
    for j in range(c.speculations):
        x = model.embedding(torch.tensor([token]))
        x = model.layers(x, torch.tensor([position + j]), cache, 0, c.exit_layer)
        states.append(x)
        token = int(early_exit(model, x)[-1].argmax())
        drafts.append(token)
        if token == c.eos:
            break
    return drafts, states


# %% [verify] Verify with the remaining layers
@torch.no_grad()
def verify(model, drafts, states, position, cache, c: Config):
    """The last draft through layers 1 … E, then x_p and every draft from layer E + 1 to the end in one pass: the
    first E layers are not run again for tokens the draft already ran. Greedy acceptance keeps the drafts up to the
    first that differs from the full model's argmax, then the full model's next token."""
    k = len(drafts)
    last = model.embedding(torch.tensor(drafts[-1:]))
    last = model.layers(last, torch.tensor([position + k]), cache, 0, c.exit_layer)
    x = torch.cat(states + [last])  # x_p, d_1 … d_k after layer E
    x = model.layers(x, torch.arange(position, position + k + 1), cache, c.exit_layer, c.n_layers)
    argmax = early_exit(model, x).argmax(dim=-1).tolist()
    kept = 0
    while kept < k and drafts[kept] == argmax[kept]:
        kept += 1
    crop(cache, position + 1 + kept)  # x_p and the kept drafts stay cached in every layer
    return drafts[:kept] + [argmax[kept]], kept


# %% [model] Self-speculative decoding
@torch.no_grad()
def self_speculative_generate(model, prompt, max_new_tokens):
    """Greedy decoding with self-speculation: each step drafts with E layers and verifies with the rest."""
    c = model.c
    cache = [None] * c.n_layers
    x = model.layers(model.embedding(prompt), torch.arange(len(prompt)), cache, 0, c.n_layers)
    out = [int(early_exit(model, x)[-1].argmax())]
    while len(out) < max_new_tokens and c.eos not in out:
        position = len(prompt) + len(out) - 1
        drafts, states = draft(model, out[-1], position, cache, c)
        emitted, _ = verify(model, drafts, states, position, cache, c)
        out += emitted
    out = out[: out.index(c.eos) + 1] if c.eos in out else out
    return out[:max_new_tokens]


@torch.no_grad()
def greedy_generate(model, prompt, max_new_tokens):
    """Plain greedy decoding through every layer: what self-speculation must reproduce."""
    c = model.c
    cache = [None] * c.n_layers
    x = model.layers(model.embedding(prompt), torch.arange(len(prompt)), cache, 0, c.n_layers)
    out = []
    while len(out) < max_new_tokens:
        out.append(int(early_exit(model, x)[-1].argmax()))
        if out[-1] == c.eos:
            break
        position = torch.tensor([len(prompt) + len(out) - 1])
        x = model.layers(model.embedding(torch.tensor(out[-1:])), position, cache, 0, c.n_layers)
    return out
`;export{e as default};
