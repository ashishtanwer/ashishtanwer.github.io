const e=`# BitNet b1.58 2B4T (microsoft/bitnet-b1.58-2B-4T), the model in the Efficiency Explainer's BitNet tab, as plain
# PyTorch blocks. It was trained with every linear layer's weights in {−1, 0, +1} times one scale per matrix and its
# inputs rounded to 8 bits per token (W1.58A8), so the released file stores the ternary values packed four to a
# byte. Its architecture is Llama-like with squared ReLU and two extra RMSNorms ("subln") before o_proj and
# down_proj. bitnet.cpp's kernels are what make it fast; on a Mac in PyTorch it computes the same numbers slowly.
# Each "# %% [id]" section is one block of the page.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # bitnet-b1.58-2B-4T
    vocab_size: int = 128256  # the Llama 3 tokenizer
    d_model: int = 2560
    n_layers: int = 30
    n_heads: int = 20  # 128 dimensions per head
    n_kv_heads: int = 5  # grouped-query attention: four query heads per key/value head
    d_mlp: int = 6912
    rope_theta: float = 500_000.0
    eps: float = 1e-5


VALUES_PER_BYTE = 4  # a ternary weight needs 2 bits


# %% [unpack] Unpack four ternary weights per byte
def unpack(packed):
    """packed (out/4, in) uint8 → ternary weights (out, in) in {−1, 0, +1}. Byte row i holds rows i, i + out/4,
    i + 2·out/4 and i + 3·out/4 of the matrix in its bit pairs 0–1, 2–3, 4–5 and 6–7, each stored as weight + 1."""
    rows = packed.shape[0]
    parts = [(packed >> (2 * k)) & 3 for k in range(VALUES_PER_BYTE)]  # each (out/4, in) in 0 … 2
    return torch.cat(parts, dim=0).to(torch.int8) - 1  # (out, in)


# %% [absmean] Ternary weights from the absmean
def absmean_quantize(W, eps=1e-5):
    """How the ternary weights are made from the full-precision weights kept during training:
    β = mean |W| over the whole matrix, T = round(W / β) clipped to {−1, 0, +1}, and W ≈ β · T."""
    beta = W.float().abs().mean().clamp(min=eps)
    T = torch.clamp(torch.round(W.float() * (1 / beta)), -1, 1)
    return T, beta


# %% [act-quant] 8-bit activations
def quantize_activations(x, bits=8):
    """Each token's input vector is scaled so its largest |value| becomes 127 and rounded: x ≈ x_q · γ / 127 with
    x_q integers in −128 … 127 and γ = max |x| of that token. Returns (x_q, 127 / γ)."""
    scale = (2 ** (bits - 1) - 1) / x.float().abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_q = (x.float() * scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return x_q, scale


# %% [bitlinear] BitLinear
class BitLinear(nn.Module):
    """A linear layer of BitNet b1.58: y = (x_q / scale) · Tᵀ · β. The checkpoint stores T packed and β as
    weight_scale; nothing else about the layer is learned."""

    def __init__(self, packed, weight_scale):
        super().__init__()
        self.register_buffer("weight", unpack(packed).float())  # (out, in) ternary
        self.register_buffer("weight_scale", weight_scale.float())  # β, shape (1,)

    def forward(self, x):  # x: (..., in)
        x_q, scale = quantize_activations(x)
        return F.linear(x_q / scale, self.weight) * self.weight_scale  # (..., out)


# %% [additions] A matrix product without multiplications
def ternary_dot(x_q, t):
    """One output of BitLinear before its two scales: with t in {−1, 0, +1} the dot product x_q · t is the sum of the
    inputs where t = +1 minus the sum where t = −1. The zeros are skipped; nothing is multiplied."""
    return x_q[..., t == 1].sum(-1) - x_q[..., t == -1].sum(-1)


def bitlinear_by_additions(x, layer):
    """Every output of a BitLinear layer from integer additions, then the two scales once per output."""
    x_q, scale = quantize_activations(x)
    sums = torch.stack([ternary_dot(x_q, t) for t in layer.weight], dim=-1)  # integers
    return sums / scale * layer.weight_scale


# %% [mlp] MLP with squared ReLU and a sub-norm
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


class MLP(nn.Module):
    """down(subnorm(relu(gate x)² · up x)): the extra RMSNorm keeps the input of down_proj, the layer whose inputs
    are rounded to 8 bits next, at a steady scale."""

    def __init__(self, layers: dict, c: Config):
        super().__init__()
        self.gate, self.up, self.down = layers["gate_proj"], layers["up_proj"], layers["down_proj"]
        self.sub_norm = RMSNorm(c.d_mlp, c.eps)

    def forward(self, x):
        return self.down(self.sub_norm(F.relu(self.gate(x)).pow(2) * self.up(x)))


# %% [attention] Attention with a sub-norm
def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    d = x.shape[-1]
    # written as transformers writes it: 8-bit rounding downstream turns last-bit differences into flipped integers
    frequencies = 1.0 / theta ** (torch.arange(0, d, 2).float() / d)
    angles = positions[:, None].float() * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Attention(nn.Module):
    """Causal grouped-query attention whose four projections are BitLinear layers; the heads' concatenated output
    is RMS-normalized before o_proj."""

    def __init__(self, layers: dict, c: Config):
        super().__init__()
        self.c = c
        self.q, self.k, self.v, self.out = (layers[n] for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.sub_norm = RMSNorm(c.d_model, c.eps)

    def forward(self, x):  # x: (batch, seq, d_model)
        c, (b, t, _) = self.c, x.shape
        q = self.q(x).view(b, t, c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, 128)
        k = self.k(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, -1).transpose(1, 2)
        positions = torch.arange(t)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        v = v.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        scores = q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5
        scores = scores + torch.full((t, t), float("-inf")).triu(1)
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
        return self.out(self.sub_norm(out))


# %% [block] Transformer block
class Block(nn.Module):
    def __init__(self, layers: dict, c: Config):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.d_model, c.eps), RMSNorm(c.d_model, c.eps)
        self.attention, self.mlp = Attention(layers, c), MLP(layers, c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [model] The whole model
class BitNet(nn.Module):
    """Embedding → 30 blocks → RMSNorm → logits. The embedding (shared with the output layer) stays in bf16:
    only the linear layers inside the blocks are ternary."""

    def __init__(self, state: dict, c: Config = Config()):
        super().__init__()
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        blocks = []
        for i in range(c.n_layers):
            prefix = f"model.layers.{i}."
            linear = lambda name, group: BitLinear(
                state[f"{prefix}{group}.{name}.weight"], state[f"{prefix}{group}.{name}.weight_scale"]
            )
            layers = {n: linear(n, "self_attn") for n in ("q_proj", "k_proj", "v_proj", "o_proj")}
            layers |= {n: linear(n, "mlp") for n in ("gate_proj", "up_proj", "down_proj")}
            blocks.append(Block(layers, c))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(c.d_model, c.eps)
        with torch.no_grad():
            self.embedding.weight.copy_(state["model.embed_tokens.weight"].float())
            self.norm.weight.copy_(state["model.norm.weight"].float())
            for i, block in enumerate(self.blocks):
                prefix = f"model.layers.{i}."
                block.norm1.weight.copy_(state[prefix + "input_layernorm.weight"].float())
                block.norm2.weight.copy_(state[prefix + "post_attention_layernorm.weight"].float())
                block.attention.sub_norm.weight.copy_(state[prefix + "self_attn.attn_sub_norm.weight"].float())
                block.mlp.sub_norm.weight.copy_(state[prefix + "mlp.ffn_sub_norm.weight"].float())

    def forward(self, ids):  # ids: (batch, seq)
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x) @ self.embedding.weight.T  # (batch, seq, vocab)
`;export{e as default};
