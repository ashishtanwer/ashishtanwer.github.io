const e=`# Kimi K3 (moonshotai/Kimi-K3), the model in the Transformer Explainer's Kimi K3 tab: 93 layers, three
# of every four with Kimi Delta Attention (a gated delta-rule linear attention with a fixed-size state)
# and every fourth with MLA (softmax attention over a 512-D latent cache, no RoPE), each followed by a
# latent mixture of 896 experts with sigmoid routing. Residuals are attention residuals: instead of
# adding every layer's output to one running sum, each layer reads a softmax mix of earlier 12-layer
# block sums. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    hidden: int = 7168
    layers: int = 93
    vocab: int = 163840
    eps: float = 1e-5
    # Kimi Delta Attention, in the 1-based layers listed here; the rest are MLA
    kda_layers: tuple = field(default_factory=lambda: tuple(i for i in range(1, 92) if i % 4))
    kda_heads: int = 96
    kda_head_dim: int = 128
    conv_kernel: int = 4
    gate_lower_bound: float = -5.0  # per-step decay stays in (e^-5, 1)
    # Multi-head latent attention
    mla_heads: int = 96
    q_rank: int = 1536
    kv_rank: int = 512
    nope_dim: int = 128  # query/key dims compared by content
    rope_dim: int = 64  # the shared per-token key part; Kimi applies no RoPE to it
    v_dim: int = 128
    # Latent MoE
    experts: int = 896
    top_k: int = 16
    latent: int = 3584  # routed experts work in a 3,584-D latent, half the hidden size
    expert_ffn: int = 3072
    shared_experts: int = 2
    dense_ffn: int = 33792  # layer 1 has a dense MLP instead of experts
    dense_layers: int = 1
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    # Attention residuals
    block: int = 12  # layers per block whose outputs are summed before later layers attend to them


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


# %% [short-conv] Short convolution
class ShortConv(nn.Module):
    """Depthwise causal convolution over the last 4 tokens, then SiLU: each channel mixes its own
    value at t-3 … t. It runs on q, k and v before the delta rule."""

    def __init__(self, channels, kernel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel))

    def forward(self, x):  # x: (batch, seq, channels)
        t, kernel = x.shape[1], self.weight.shape[-1]
        y = F.conv1d(x.transpose(1, 2), self.weight, padding=kernel - 1, groups=x.shape[-1])
        return F.silu(y[..., :t].transpose(1, 2))  # keep outputs 0 … t-1: no future tokens


# %% [kda-gate] Forget gate α and write strength β
def kda_gates(f, b, A_log, dt_bias, lower_bound):
    """f: (batch, seq, heads, 128) from the low-rank f_a → f_b projection; b: (batch, seq, heads).
    Each head gets its own decay for each of its 128 key channels (fine-grained), bounded below:
    log α = lower_bound · σ(e^A · (f + dt_bias)) ∈ (lower_bound, 0). β = σ(b) ∈ (0, 1)."""
    heads = f.shape[-2]
    log_alpha = lower_bound * torch.sigmoid(A_log.exp()[:, None] * (f + dt_bias.view(heads, -1)))
    return log_alpha, torch.sigmoid(b)


# %% [delta-rule] Gated delta rule
def delta_rule(q, k, v, log_alpha, beta, state=None):
    """q, k, v, log_alpha: (batch, seq, heads, 128); beta: (batch, seq, heads); q and k are unit
    vectors. Each head keeps one 128 × 128 state S (value × key) and, token by token,
        S ← S · diag(α) · (I − β k kᵀ) + β v kᵀ,     o = S q / √128.
    diag(α) forgets per key channel; the β k kᵀ term erases what S held along k before writing v,
    so a repeated key overwrites its value instead of piling up."""
    b, t, h, d = k.shape
    S = q.new_zeros(b, h, v.shape[-1], d) if state is None else state
    out = []
    for i in range(t):
        k_i, v_i, a_i, b_i = k[:, i], v[:, i], log_alpha[:, i].exp(), beta[:, i, :, None]
        S = S * a_i[:, :, None, :]  # forget, one rate per key channel
        old = (S @ k_i[..., None])[..., 0]  # what S currently returns for k_i
        S = S + (b_i * (v_i - old))[..., None] * k_i[:, :, None, :]  # the delta-rule write
        out.append((S @ q[:, i, :, :, None])[..., 0] / math.sqrt(d))
    return torch.stack(out, dim=1), S


# %% [kda] Kimi Delta Attention
def l2norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


class KimiDeltaAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        width = c.kda_heads * c.kda_head_dim
        self.q_proj = nn.Linear(c.hidden, width, bias=False)
        self.k_proj = nn.Linear(c.hidden, width, bias=False)
        self.v_proj = nn.Linear(c.hidden, width, bias=False)
        self.q_conv1d = ShortConv(width, c.conv_kernel)
        self.k_conv1d = ShortConv(width, c.conv_kernel)
        self.v_conv1d = ShortConv(width, c.conv_kernel)
        self.f_a_proj = nn.Linear(c.hidden, c.kda_head_dim, bias=False)  # low rank: 7168 → 128 → 12288
        self.f_b_proj = nn.Linear(c.kda_head_dim, width, bias=False)
        self.b_proj = nn.Linear(c.hidden, c.kda_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(c.kda_heads))
        self.dt_bias = nn.Parameter(torch.zeros(width))
        self.g_proj = nn.Linear(c.hidden, width, bias=False)  # full-rank output gate
        self.o_norm = RMSNorm(c.kda_head_dim, c.eps)
        self.o_proj = nn.Linear(width, c.hidden, bias=False)

    def forward(self, x):  # x: (batch, seq, hidden), after the input RMSNorm
        heads = lambda y: y.unflatten(-1, (self.c.kda_heads, self.c.kda_head_dim))  # noqa: E731
        q = l2norm(heads(self.q_conv1d(self.q_proj(x))))  # unit-length queries and keys
        k = l2norm(heads(self.k_conv1d(self.k_proj(x))))
        v = heads(self.v_conv1d(self.v_proj(x)))
        log_alpha, beta = kda_gates(
            heads(self.f_b_proj(self.f_a_proj(x))), self.b_proj(x), self.A_log, self.dt_bias, self.c.gate_lower_bound
        )
        o, _ = delta_rule(q, k, v, log_alpha, beta)
        o = self.o_norm(o) * torch.sigmoid(heads(self.g_proj(x)))  # per-head RMSNorm, sigmoid gate
        return self.o_proj(o.flatten(-2))


# %% [mla] Multi-head latent attention
class MLA(nn.Module):
    """DeepSeek-style MLA with two changes: no RoPE anywhere (position comes from the KDA layers'
    recurrence) and a sigmoid output gate. The cache keeps 512 + 64 numbers per token."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        h, qk = c.mla_heads, c.nope_dim + c.rope_dim
        self.q_a_proj = nn.Linear(c.hidden, c.q_rank, bias=False)
        self.q_a_layernorm = RMSNorm(c.q_rank, 1e-6)
        self.q_b_proj = nn.Linear(c.q_rank, h * qk, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(c.hidden, c.kv_rank + c.rope_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(c.kv_rank, 1e-6)
        self.kv_b_proj = nn.Linear(c.kv_rank, h * (c.nope_dim + c.v_dim), bias=False)
        self.g_proj = nn.Linear(c.hidden, h * c.v_dim, bias=False)
        self.o_proj = nn.Linear(h * c.v_dim, c.hidden, bias=False)

    def forward(self, x):  # x: (batch, seq, hidden), after the input RMSNorm
        c, (b, t, _) = self.c, x.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(b, t, c.mla_heads, -1)
        latent, k_shared = self.kv_a_proj_with_mqa(x).split([c.kv_rank, c.rope_dim], dim=-1)  # the cache
        k_nope, v = self.kv_b_proj(self.kv_a_layernorm(latent)).view(b, t, c.mla_heads, -1).split(
            [c.nope_dim, c.v_dim], dim=-1
        )
        k = torch.cat([k_nope, k_shared[:, :, None].expand(-1, -1, c.mla_heads, -1)], dim=-1)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(q.shape[-1])
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        o = torch.einsum("bhqk,bkhd->bqhd", weights, v).flatten(-2)
        return self.o_proj(o * torch.sigmoid(self.g_proj(x)))


# %% [situ] SiTU gated activation
def situ(gate, up, beta=4.0, linear_beta=25.0):
    """Moonshot's soft-capped SwiGLU: β·tanh(gate/β)·σ(gate) in place of SiLU, and the up branch
    soft-capped at ±25 the same way."""
    return beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * (linear_beta * torch.tanh(up / linear_beta))


class MLP(nn.Module):
    def __init__(self, dim, ffn, c: Config):
        super().__init__()
        self.c = c
        self.gate_proj = nn.Linear(dim, ffn, bias=False)
        self.up_proj = nn.Linear(dim, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, dim, bias=False)

    def forward(self, x):
        return self.down_proj(situ(self.gate_proj(x), self.up_proj(x), self.c.situ_beta, self.c.situ_linear_beta))


# %% [mxfp4] MXFP4 expert weights
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def mxfp4(packed, scale):
    """Routed experts ship as MXFP4: two 4-bit floats (1 sign, 2 exponent, 1 mantissa bits) per byte,
    low nibble first, and one power-of-two scale 2^(s - 127) per 32 weights of a row."""
    nibbles = torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2).long()
    values = E2M1[nibbles & 7] * torch.where(nibbles >= 8, -1.0, 1.0)
    return values * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)


# %% [router] Sigmoid router
class Router(nn.Module):
    """Every expert gets an independent sigmoid score. A per-expert bias, tuned during training to
    balance load, decides which 16 are picked but not how much each counts; the picked scores are
    renormalized to sum to 1."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.weight = nn.Parameter(torch.zeros(c.experts, c.hidden))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(c.experts))

    def forward(self, x):  # x: (tokens, hidden)
        scores = torch.sigmoid(x @ self.weight.T)
        picked = (scores + self.e_score_correction_bias).topk(self.c.top_k, dim=-1).indices
        weights = scores.gather(-1, picked)
        return picked, weights / (weights.sum(-1, keepdim=True) + 1e-20)


# %% [moe] Latent mixture of experts
class Expert(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.w1 = nn.Linear(c.latent, c.expert_ffn, bias=False)  # gate
        self.w3 = nn.Linear(c.latent, c.expert_ffn, bias=False)  # up
        self.w2 = nn.Linear(c.expert_ffn, c.latent, bias=False)  # down

    def forward(self, x):
        return self.w2(situ(self.w1(x), self.w3(x), self.c.situ_beta, self.c.situ_linear_beta))


class LatentMoE(nn.Module):
    """Routed experts run in a 3,584-D latent: one shared projection down, 16 of 896 experts there,
    an RMSNorm and one projection back up. Two always-on shared experts read the full 7,168-D input."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = Router(c)
        self.experts = nn.ModuleList(Expert(c) for _ in range(c.experts))
        self.routed_expert_down_proj = nn.Linear(c.hidden, c.latent, bias=False)
        self.routed_expert_norm = RMSNorm(c.latent, c.eps)
        self.routed_expert_up_proj = nn.Linear(c.latent, c.hidden, bias=False)
        self.shared_experts = MLP(c.hidden, c.expert_ffn * c.shared_experts, c)

    def forward(self, x):  # x: (batch, seq, hidden), after the post-attention RMSNorm
        flat = x.flatten(0, -2)
        picked, weights = self.gate(flat)
        latent = self.routed_expert_down_proj(flat)
        routed = torch.zeros_like(latent)
        for slot in range(picked.shape[-1]):
            for e in picked[:, slot].unique().tolist():
                rows = picked[:, slot] == e
                routed[rows] += weights[rows, slot, None] * self.experts[e](latent[rows])
        routed = self.routed_expert_up_proj(self.routed_expert_norm(routed))
        return routed.view_as(x) + self.shared_experts(x)


# %% [attn-res] Attention residuals
def attn_res(candidates, query, eps):
    """candidates: (tokens, n, hidden), the completed 12-layer block sums followed by the current
    block's partial sum; query: (hidden,), a learned vector per sublayer. Instead of reading the
    plain running sum, a sublayer reads a softmax mix of them, scored on their RMS-normalized form."""
    keys = candidates * torch.rsqrt(candidates.pow(2).mean(-1, keepdim=True) + eps)
    weights = (keys @ query).softmax(dim=-1)  # (tokens, n)
    return (weights[:, None] @ candidates)[:, 0], weights


# %% [block] Decoder layer
class Layer(nn.Module):
    def __init__(self, c: Config, index: int):
        super().__init__()
        self.c, self.index = c, index
        self.self_attn = KimiDeltaAttention(c) if index + 1 in c.kda_layers else MLA(c)
        self.mlp = MLP(c.hidden, c.dense_ffn, c) if index < c.dense_layers else LatentMoE(c)
        self.input_layernorm = RMSNorm(c.hidden, c.eps)
        self.post_attention_layernorm = RMSNorm(c.hidden, c.eps)
        # each sublayer's attention-residual query is the product of a norm gain and a 1-row projection
        self.self_attention_res_norm = RMSNorm(c.hidden, c.eps)
        self.self_attention_res_proj = nn.Linear(c.hidden, 1, bias=False)
        self.mlp_res_norm = RMSNorm(c.hidden, c.eps)
        self.mlp_res_proj = nn.Linear(c.hidden, 1, bias=False)

    def query(self, norm, proj):
        return norm.weight * proj.weight[0]

    def forward(self, partial, blocks):
        """partial: (tokens, hidden), the sum of this block's sublayer outputs so far (the embedding
        before layer 1); blocks: (tokens, n, hidden), the sums of completed blocks."""
        shape = (1, -1, self.c.hidden)
        x = partial
        if blocks.shape[1]:
            query = self.query(self.self_attention_res_norm, self.self_attention_res_proj)
            x, _ = attn_res(torch.cat([blocks, partial[:, None]], dim=1), query, self.c.eps)
        if self.index % self.c.block == 0:  # a block ends: its sum becomes a candidate of its own
            blocks, partial = torch.cat([blocks, partial[:, None]], dim=1), None
        h = self.self_attn(self.input_layernorm(x).view(shape)).view(-1, self.c.hidden)
        partial = h if partial is None else partial + h
        query = self.query(self.mlp_res_norm, self.mlp_res_proj)
        x, _ = attn_res(torch.cat([blocks, partial[:, None]], dim=1), query, self.c.eps)
        h = self.mlp(self.post_attention_layernorm(x).view(shape)).view(-1, self.c.hidden)
        return partial + h, blocks


# %% [model] Kimi K3 language model
class KimiK3(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab, c.hidden)
        self.layers = nn.ModuleList(Layer(c, i) for i in range(c.layers))
        self.output_attn_res_norm = RMSNorm(c.hidden, c.eps)
        self.output_attn_res_proj = nn.Linear(c.hidden, 1, bias=False)
        self.norm = RMSNorm(c.hidden, c.eps)
        self.lm_head = nn.Linear(c.hidden, c.vocab, bias=False)

    def forward(self, ids):  # ids: (seq,), one sequence
        partial = self.embed_tokens(ids)
        blocks = partial.new_zeros(len(ids), 0, self.c.hidden)
        for layer in self.layers:
            partial, blocks = layer(partial, blocks)
        query = self.output_attn_res_norm.weight * self.output_attn_res_proj.weight[0]
        x, _ = attn_res(torch.cat([blocks, partial[:, None]], dim=1), query, self.c.eps)
        return self.lm_head(self.norm(x))  # (seq, vocab) next-token logits
`;export{e as default};
