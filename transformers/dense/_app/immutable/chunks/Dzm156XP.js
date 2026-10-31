const e=`# gpt-oss-20b, the model in the gpt-oss tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    vocab_size: int = 201088  # o200k_harmony
    d_model: int = 2880
    n_layers: int = 24  # sliding-window and full attention in turn, starting with sliding
    n_heads: int = 64
    n_kv_heads: int = 8  # grouped-query attention: eight query heads share each key/value head
    head_dim: int = 64
    sliding_window: int = 128  # a sliding block's query i sees keys i − 127 … i
    n_experts: int = 32
    top_k: int = 4  # experts each token is routed to
    d_expert: int = 2880
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    rope_theta: float = 150_000.0
    rope_factor: float = 32.0  # YaRN: trained on 4,096 positions, stretched 32× to 131,072
    rope_original_context: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    eps: float = 1e-5

    def kind(self, layer: int) -> str:
        return "sliding" if layer % 2 == 0 else "full"


# %% [harmony] Harmony chat format
def harmony(messages, reasoning="medium", date="2026-10-30"):
    """gpt-oss reads a conversation as messages wrapped in special tokens, <|start|>role<|message|>text<|end|>.
    The system message sets the reasoning effort and the channels. The prompt ends with <|start|>assistant; the
    model then writes <|channel|>analysis<|message|>… (its reasoning) <|end|>, and
    <|start|>assistant<|channel|>final<|message|>… (the answer) <|return|>."""
    system = (
        "You are ChatGPT, a large language model trained by OpenAI.\\nKnowledge cutoff: 2024-06\\n"
        f"Current date: {date}\\n\\nReasoning: {reasoning}\\n\\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    text = f"<|start|>system<|message|>{system}<|end|>"
    for message in messages:
        text += f"<|start|>{message['role']}<|message|>{message['content']}<|end|>"
    return text + "<|start|>assistant"


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [embedding] Embedding
class Embedding(nn.Module):
    """A plain token lookup. Positions enter later, through RoPE inside attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids)  # (batch, seq, d_model)


# %% [rope] RoPE with YaRN
def yarn(c: Config):
    """The 32 rotation frequencies of a 64-D head, and the factor cos and sin are scaled by. Pairs that turn
    more than 32 times over the 4,096 trained positions keep their frequency, pairs that turn less than once
    are slowed 32×, and the pairs in between are blended."""
    d = c.head_dim
    wavelengths = c.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)

    def pair(turns):  # the (fractional) pair index that turns \`turns\` times over the trained context
        return d * math.log(c.rope_original_context / (turns * 2 * math.pi)) / (2 * math.log(c.rope_theta))

    low, high = max(pair(c.rope_beta_fast), 0), min(pair(c.rope_beta_slow), d - 1)
    slowed = ((torch.arange(d // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    frequencies = slowed / (c.rope_factor * wavelengths) + (1 - slowed) / wavelengths
    return frequencies, 0.1 * math.log(c.rope_factor) + 1  # (d/2,), 1.347


def rope(x, positions, c: Config):
    """Rotate each pair (i, i + d/2) of every head vector by position × frequency_i, scaled by YaRN's factor."""
    frequencies, scale = yarn(c)
    angles = positions[:, None].float() * frequencies.to(x.device)  # (seq, d/2)
    cos, sin = angles.cos() * scale, angles.sin() * scale
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


# %% [sliding-window] Sliding and full masks
def visible(queries, keys, kind, window):
    """Which keys each query may attend to: every earlier key in a full block, only the last \`window\`
    (itself included) in a sliding one. queries, keys: positions → (queries, keys) booleans."""
    distance = queries[:, None] - keys[None, :]
    if kind == "full":
        return distance >= 0
    return (distance >= 0) & (distance < window)


# %% [sinks] Learned attention sinks
def attend_with_sink(scores, sink):
    """Softmax over the visible keys plus one more logit per head that belongs to no token. The sink takes
    its share of the denominator and is then dropped, so a head can attend to (almost) nothing.
    scores: (batch, heads, queries, keys), -inf where hidden; sink: (heads,).
    Returns the key weights, which sum to 1 − the sink's share, and that share."""
    extra = sink.view(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
    probabilities = torch.cat((scores, extra), dim=-1).softmax(dim=-1)
    return probabilities[..., :-1], probabilities[..., -1]


# %% [attention] Self-attention with sinks
class Attention(nn.Module):
    """Grouped-query attention with biases on every projection and a learned sink logit per head."""

    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.c, self.kind = c, c.kind(layer)
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model)
        self.sinks = nn.Parameter(torch.zeros(c.n_heads))

    def forward(self, x, start=0):  # x: (batch, seq, d_model), starting at position \`start\`
        c, (b, t, _) = self.c, x.shape
        positions = torch.arange(start, start + t, device=x.device)
        q = self.q(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        k = self.k(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, positions, c), rope(k, positions, c)
        group = c.n_heads // c.n_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-2, -1) / c.head_dim**0.5
        mask = visible(positions, positions, self.kind, c.sliding_window)
        weights, _ = attend_with_sink(scores.masked_fill(~mask, float("-inf")), self.sinks)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))


# %% [mxfp4] MXFP4 weights
FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])  # E2M1 codes 0–15


def dequantize(blocks, scales):
    """MXFP4: every 32 weights share one 8-bit power of two (E8M0: 2^(s − 127)), and each weight is a 4-bit
    float (E2M1: sign, 2 exponent bits, 1 mantissa bit), two per byte, low nibble first.
    blocks: (..., rows, groups, 16) uint8, scales: (..., rows, groups) uint8 → (..., rows, groups × 32)."""
    codes = torch.stack((blocks & 0x0F, blocks >> 4), dim=-1).flatten(-2)  # (..., rows, groups, 32)
    values = FP4[codes.long()] * torch.exp2(scales.float() - 127)[..., None]
    return values.flatten(-2)


# %% [router] Top-4 router
class Router(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.linear = nn.Linear(c.d_model, c.n_experts)
        self.top_k = c.top_k

    def forward(self, x):  # x: (tokens, d_model)
        top_logits, experts = self.linear(x).topk(self.top_k, dim=-1)
        return experts, top_logits.softmax(dim=-1)  # weights over the four chosen experts only


# %% [experts] Experts
class Experts(nn.Module):
    """32 SwiGLU MLPs, stored in MXFP4. gpt-oss's SwiGLU clamps the gate below 7 and the up half to ±7,
    uses gate · sigmoid(1.702 · gate) and adds 1 to the up half. Gate and up rows are interleaved."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate_up = nn.Parameter(torch.zeros(c.n_experts, 2 * c.d_expert, c.d_model))
        self.gate_up_bias = nn.Parameter(torch.zeros(c.n_experts, 2 * c.d_expert))
        self.down = nn.Parameter(torch.zeros(c.n_experts, c.d_model, c.d_expert))
        self.down_bias = nn.Parameter(torch.zeros(c.n_experts, c.d_model))
        self.alpha, self.limit = c.swiglu_alpha, c.swiglu_limit

    def forward(self, e, x):  # expert e on x: (tokens, d_model)
        h = x @ self.gate_up[e].T + self.gate_up_bias[e]  # (tokens, 2 × d_expert)
        gate = h[:, 0::2].clamp(max=self.limit)
        up = h[:, 1::2].clamp(-self.limit, self.limit)
        return (gate * torch.sigmoid(self.alpha * gate) * (up + 1)) @ self.down[e].T + self.down_bias[e]

    @torch.no_grad()
    def load_mxfp4(self, gate_up_blocks, gate_up_scales, down_blocks, down_scales):
        for e in range(self.gate_up.shape[0]):  # one expert at a time keeps the 4-bit codes' memory small
            self.gate_up[e] = dequantize(gate_up_blocks[e], gate_up_scales[e])
            self.down[e] = dequantize(down_blocks[e], down_scales[e])


# %% [mlp] Mixture of experts
class MoE(nn.Module):
    """The router picks four of 32 experts per token; their outputs are summed with the router's weights."""

    def __init__(self, c: Config):
        super().__init__()
        self.router = Router(c)
        self.experts = Experts(c)

    def forward(self, x):  # x: (batch, seq, d_model)
        tokens = x.reshape(-1, x.shape[-1])
        experts, gates = self.router(tokens)  # (tokens, 4) each
        out = torch.zeros_like(tokens)
        for e in experts.unique().tolist():
            rows, slot = (experts == e).nonzero(as_tuple=True)
            out.index_add_(0, rows, gates[rows, slot, None] * self.experts(e, tokens[rows]))
        return out.view(x.shape)


# %% [block] Transformer block
class Block(nn.Module):
    def __init__(self, c: Config, layer: int):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c, layer)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.moe = MoE(c)

    def forward(self, x, start=0):
        x = x + self.attention(self.norm1(x), start)
        return x + self.moe(self.norm2(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm and the vocabulary projection, which is not tied to the embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(x))  # logits: (batch, seq, vocab)


# %% [model] Full model
class GptOss(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c, layer) for layer in range(c.n_layers))
        self.output = Output(c)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# ids = tokenizer(harmony([{"role": "user", "content": "…"}]))["input_ids"]
# next_token = GptOss()(torch.tensor([ids]))[0, -1].argmax()
`;export{e as default};
