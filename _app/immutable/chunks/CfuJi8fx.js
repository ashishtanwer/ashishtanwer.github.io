const e=`# Sesame CSM-1B (Conversational Speech Model, Sesame 2025), sesame/csm-1b, as plain PyTorch. CSM speaks in Mimi codes
# (the Tokenization Explainer's Mimi tab): 12.5 frames a second, 32 codes per frame, each from a codebook of 2,048
# vectors. A conversation is one sequence: each turn's text (Llama 3's tokenizer, "[speaker]" then the words) and
# then its audio frames. A Llama-style backbone (16 layers, 2,048 wide) reads it and predicts codebook 0 of the next
# frame; a depth decoder (4 layers, 1,024 wide) starts from the backbone's last state and predicts codebooks 1 to 31
# one after another. The frame's 32 codes are embedded, added up and appended, and the loop runs until a frame whose
# codes are all 0; Mimi's decoder turns the frames into 24 kHz audio.
# Each "# %% [id]" section is one step of the CSM tab's diagram.
import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    hidden: int = 2048
    layers: int = 16
    heads: int = 32
    kv_heads: int = 8  # grouped-query attention: 4 query heads share each key/value head
    head_dim: int = 64
    ffn: int = 8192
    rope: tuple = (500000.0, 32.0, 0.125, 0.5, 1024)  # θ, factor, low and high frequency factors, original context
    eps: float = 1e-5
    text_vocab: int = 128256
    codebooks: int = 32
    codebook_vocab: int = 2051  # 2,048 codes and 3 spare ids; a frame of 0s ends the turn
    depth_hidden: int = 1024
    depth_layers: int = 4
    depth_heads: int = 8
    depth_kv_heads: int = 2
    depth_head_dim: int = 128
    depth_ffn: int = 8192
    depth_rope: tuple = (500000.0, 32.0, 0.001953125, 0.0078125, 16)
    temperature: float = 0.9
    top_k: int = 50


# %% [rope] Rotary positions with Llama 3's scaling
def rope_frequencies(head_dim, rope):
    """One rotation speed per pair of dimensions. Llama 3's scaling slows the slow pairs (wavelengths longer than
    context / low) by \`factor\`, keeps the fast ones and blends between them."""
    theta, factor, low, high, context = rope
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    wavelength = 2 * math.pi / inv
    scaled = torch.where(wavelength > context / low, inv / factor, inv)
    smooth = (context / wavelength - low) / (high - low)
    blended = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelength < context / high) * ~(wavelength > context / low)
    return torch.where(medium, blended, scaled)


def rotate(x, inv):
    """x (1, heads, T, d): rotate each pair (i, i + d/2) at position t by t × its speed."""
    angles = torch.arange(x.shape[2]).float()[:, None] * inv
    cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
    first, second = x.chunk(2, -1)
    return x * cos + torch.cat([-second, first], -1) * sin


# %% [attention] Causal attention, grouped query heads
class Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, rope):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.register_buffer("inv", rope_frequencies(head_dim, rope), persistent=False)

    def forward(self, x):
        """x (1, T, hidden) → output (1, T, hidden) and weights (heads, T, T): row t is position t's attention."""
        T, d = x.shape[1], self.head_dim
        q = rotate(self.q_proj(x).view(1, T, self.heads, d).transpose(1, 2), self.inv)
        k = rotate(self.k_proj(x).view(1, T, self.kv_heads, d).transpose(1, 2), self.inv)
        v = self.v_proj(x).view(1, T, self.kv_heads, d).transpose(1, 2)
        group = self.heads // self.kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = (q @ k.transpose(-1, -2)) * d**-0.5 + torch.full((T, T), float("-inf")).triu(1)
        weights = torch.softmax(scores, dim=-1)
        return self.o_proj((weights @ v).transpose(1, 2).reshape(1, T, -1)), weights[0]


# %% [mlp] SwiGLU feed-forward
class MLP(nn.Module):
    def __init__(self, hidden, ffn):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


# %% [block] One Llama block
class Block(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, ffn, rope, eps):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.self_attn = Attention(hidden, heads, kv_heads, head_dim, rope)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.mlp = MLP(hidden, ffn)

    def forward(self, x):
        attended, weights = self.self_attn(self.input_layernorm(x))
        x = x + attended
        return x + self.mlp(self.post_attention_layernorm(x)), weights


class Transformer(nn.Module):
    """A stack of blocks and a final RMSNorm; the backbone and the depth decoder are both one of these."""

    def __init__(self, hidden, layers, heads, kv_heads, head_dim, ffn, rope, eps):
        super().__init__()
        self.layers = nn.ModuleList(Block(hidden, heads, kv_heads, head_dim, ffn, rope, eps) for _ in range(layers))
        self.norm = RMSNorm(hidden, eps)

    def forward(self, x):
        """x (1, T, hidden) → normalised states (1, T, hidden) and every layer's attention weights."""
        weights = []
        for layer in self.layers:
            x, w = layer(x)
            weights.append(w)
        return self.norm(x), weights


# %% [embedding] Text tokens and audio frames, one sequence
class CSM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed_text_tokens = nn.Embedding(c.text_vocab, c.hidden)
        # One table for all 32 codebooks: codebook k's code i is row k · 2,051 + i. The depth decoder shares it.
        self.embed_audio_tokens = nn.Embedding(c.codebooks * c.codebook_vocab, c.hidden)
        self.backbone = Transformer(c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.ffn, c.rope, c.eps)
        self.lm_head = nn.Linear(c.hidden, c.codebook_vocab, bias=False)  # codebook 0
        self.depth_projector = nn.Linear(c.hidden, c.depth_hidden, bias=False)
        self.depth = Transformer(c.depth_hidden, c.depth_layers, c.depth_heads, c.depth_kv_heads, c.depth_head_dim,
                                 c.depth_ffn, c.depth_rope, c.eps)
        self.codebooks_head = nn.Parameter(torch.zeros(c.codebooks - 1, c.depth_hidden, c.codebook_vocab))


def embed(m: CSM, text_ids, frames, is_audio):
    """text_ids (T,), frames (T, 32), is_audio (T,) → (1, T, 2048). A text position is its token's row; an audio
    position is the sum of its 32 codes' rows. The end of a turn's audio is a frame of 0s."""
    text = m.embed_text_tokens(text_ids)
    audio = m.embed_audio_tokens(frames + torch.arange(m.c.codebooks) * m.c.codebook_vocab).sum(1)
    return torch.where(is_audio[:, None], audio, text)[None]


# %% [backbone] The backbone: codebook 0 of the next frame
def backbone(m: CSM, x):
    """x (1, T, 2048) → the last position's state (2048,), its codebook-0 logits (2051,) and the attention weights."""
    states, weights = m.backbone(x)
    state = states[0, -1]
    return state, m.lm_head(state), weights


# %% [sampling] Top-k sampling at temperature 0.9
def top_k(logits, c: Config):
    """logits (2051,) → the k most likely ids and their probabilities after dividing the logits by the temperature."""
    values, ids = (logits / c.temperature).topk(c.top_k)
    return ids, torch.softmax(values, dim=-1)


def sample(logits, u, c: Config):
    """The id where the cumulative probability (most likely first) passes u ∈ [0, 1); u = None takes the top one."""
    ids, probs = top_k(logits, c)
    if u is None:
        return ids[0]
    return ids[min(int((probs.cumsum(0) <= u).sum()), c.top_k - 1)]


# %% [depth-decoder] The depth decoder: codebooks 1 to 31
def depth_logits(m: CSM, state, codes):
    """state (2048,), codes (k,) = codebooks 0..k−1 of this frame → logits (k, 2051) for codebooks 1..k. Position 0
    holds the backbone's state, position j the embedding of codebook j−1's code (the backbone's table, projected to
    1,024); each codebook has its own output matrix."""
    rows = m.embed_audio_tokens(codes + torch.arange(codes.shape[0]) * m.c.codebook_vocab)
    x = m.depth_projector(torch.cat([state[None], rows])[None])
    states, _ = m.depth(x)
    return torch.stack([states[0, j] @ m.codebooks_head[j - 1] for j in range(1, codes.shape[0] + 1)])


# %% [frame] One frame: 1 + 31 codes
def frame(m: CSM, x, uniforms=None):
    """x (1, T, 2048), the conversation so far → this frame's 32 codes. uniforms (32,) drives the sampling; None
    takes the most likely code each time."""
    pick = lambda logits, k: sample(logits, None if uniforms is None else float(uniforms[k]), m.c)  # noqa: E731
    state, logits, _ = backbone(m, x)
    codes = pick(logits, 0)[None]
    for k in range(1, m.c.codebooks):
        codes = torch.cat([codes, pick(depth_logits(m, state, codes)[-1], k)[None]])
    return codes


# %% [model] A conversation in, the next turn's frames out
def generate(m: CSM, text_ids, frames, is_audio, uniforms=None, max_frames=125):
    """The conversation (text_ids, frames, is_audio as in embed) → generated frames (n, 32), stopping at a frame
    whose codebooks 0–30 are all 0. Each new frame is appended as one more audio position."""
    out = []
    for t in range(max_frames):
        codes = frame(m, embed(m, text_ids, frames, is_audio), None if uniforms is None else uniforms[t])
        if (codes[:-1] == 0).all():
            break
        out.append(codes)
        text_ids = torch.cat([text_ids, text_ids.new_zeros(1)])
        frames = torch.cat([frames, codes[None]])
        is_audio = torch.cat([is_audio, is_audio.new_ones(1)])
    return torch.stack(out) if out else frames.new_zeros(0, m.c.codebooks)


# transformers' CsmForConditionalGeneration names → this file's; Mimi (codec_model) is the Mimi tab's.
RENAMES = [
    (r"^backbone_model\\.embed_tokens\\.embed_audio_tokens", "embed_audio_tokens"),
    (r"^backbone_model\\.", "backbone."),
    (r"^depth_decoder\\.model\\.inputs_embeds_projector", "depth_projector"),
    (r"^depth_decoder\\.model\\.", "depth."),
    (r"^depth_decoder\\.codebooks_head\\.weight", "codebooks_head"),
]


def load_state(state):
    """A CsmForConditionalGeneration state dict → this file's (the tied depth-decoder table and Mimi dropped)."""
    out = {}
    for key, value in state.items():
        if key.startswith("codec_model.") or key == "depth_decoder.model.embed_tokens.weight":
            continue
        for pattern, replacement in RENAMES:
            key = re.sub(pattern, replacement, key)
        out[key] = value
    return out
`;export{e as default};
