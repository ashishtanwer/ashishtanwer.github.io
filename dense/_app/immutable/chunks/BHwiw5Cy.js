const e=`# Zamba2-1.2B, the model in the Mamba hybrid tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram. 32 of the 38 blocks are Mamba2 blocks;
# blocks 6, 12, ... 36 first run one shared transformer, then a Mamba2 block.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32000
    d_model: int = 2048
    n_layers: int = 38
    hybrid_layers: tuple = (5, 11, 17, 23, 29, 35)  # 0-based: blocks 6, 12, ... 36
    # Mamba2 mixer
    d_inner: int = 4096  # 2 × d_model
    n_mamba_heads: int = 64  # 4096 / 64 = 64 dimensions per head
    d_state: int = 128  # B and C are shared by every head (one group)
    d_conv: int = 4
    dt_min: float = 0.001  # Δ never drops below this
    # Shared transformer, over concat(x, embedding): 2 × d_model = 4096 wide
    n_heads: int = 32  # 128 dimensions per head
    d_mlp: int = 8192
    adapter_rank: int = 128
    rope_theta: float = 10000.0
    eps: float = 1e-5


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
    """Token lookup. Every hybrid block reads this embedding again, beside the residual stream."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids)  # (batch, seq, d_model)


# %% [rope] RoPE
def rope(x, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    seq, d = x.shape[-2], x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(seq, device=x.device)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Multi-head self-attention (shared)
class Adapter(nn.Module):
    """A low-rank update d_in → rank → d_out that one hybrid block adds to a shared projection."""

    def __init__(self, d_in: int, rank: int, d_out: int):
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)

    def forward(self, x):
        return self.up(self.down(x))


class Attention(nn.Module):
    """Causal attention over concat(x, embedding). The weights are shared by all six hybrid blocks;
    each block adds its own adapter to Q, K and V."""

    def __init__(self, c: Config):
        super().__init__()
        d = 2 * c.d_model
        self.c = c
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, c.d_model, bias=False)
        n = len(c.hybrid_layers)
        self.q_adapters = nn.ModuleList(Adapter(d, c.adapter_rank, d) for _ in range(n))
        self.k_adapters = nn.ModuleList(Adapter(d, c.adapter_rank, d) for _ in range(n))
        self.v_adapters = nn.ModuleList(Adapter(d, c.adapter_rank, d) for _ in range(n))

    def forward(self, x, adapter):  # x: (batch, seq, 2 × d_model); adapter: which hybrid block
        b, t, _ = x.shape
        heads = lambda z: z.view(b, t, self.c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, 128)
        q = heads(self.q(x) + self.q_adapters[adapter](x))
        k = heads(self.k(x) + self.k_adapters[adapter](x))
        v = heads(self.v(x) + self.v_adapters[adapter](x))
        q, k = rope(q, self.c.rope_theta), rope(k, self.c.rope_theta)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1] / 2)  # Zamba2 divides by √(head_dim / 2)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, -1))  # (batch, seq, d_model)


# %% [mlp] Gated MLP
class MLP(nn.Module):
    """GELU(gate) ⊙ up, shared like the attention, with each hybrid block's adapter on gate and up."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate_up = nn.Linear(c.d_model, 2 * c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)
        n = len(c.hybrid_layers)
        self.adapters = nn.ModuleList(Adapter(c.d_model, c.adapter_rank, 2 * c.d_mlp) for _ in range(n))

    def forward(self, x, adapter):
        gate, up = (self.gate_up(x) + self.adapters[adapter](x)).chunk(2, dim=-1)
        return self.down(F.gelu(gate) * up)


# %% [shared-transformer] Shared transformer
class SharedTransformer(nn.Module):
    """The one transformer block that all six hybrid blocks reuse. It reads the residual stream next
    to the original embedding and has no residual of its own: its output feeds the next Mamba mixer."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(2 * c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, embedding, adapter):
        h = self.attention(self.norm1(torch.cat([x, embedding], dim=-1)), adapter)
        return self.mlp(self.norm2(h), adapter)


# %% [conv1d] Causal conv1d
def causal_conv1d(x, weight, bias):
    """Each channel mixes its own last d_conv tokens, then SiLU. x: (batch, seq, channels);
    weight: (channels, 1, d_conv) as nn.Conv1d stores it."""
    t, width = x.shape[1], weight.shape[-1]
    padded = F.pad(x, (0, 0, width - 1, 0))  # d_conv − 1 zero tokens before the first
    out = sum(padded[:, j : j + t] * weight[:, 0, j] for j in range(width)) + bias
    return F.silu(out)


# %% [selective-scan] Selective scan
def selective_scan(x, dt, A, B, C, D):
    """Mamba2's state space recurrence, one token at a time. Head h keeps a (head_dim × d_state) state:

        S_t = exp(Δ_t A_h) S_{t-1} + Δ_t x_t B_tᵀ,    y_t = S_t C_t + D_h x_t

    x: (batch, seq, heads, head_dim); dt: (batch, seq, heads); A, D: (heads,); B, C: (batch, seq, d_state)."""
    b, t, h, p = x.shape
    S = x.new_zeros(b, h, p, B.shape[-1])
    ys = []
    for i in range(t):
        decay = torch.exp(dt[:, i] * A)[..., None, None]  # how much of each head's state survives
        S = decay * S + (dt[:, i, :, None] * x[:, i])[..., None] * B[:, i, None, None, :]
        ys.append((S @ C[:, i, None, :, None]).squeeze(-1) + D[:, None] * x[:, i])
    return torch.stack(ys, dim=1)  # (batch, seq, heads, head_dim)


# %% [mixing-matrix] State-space mixing
def mixing_matrix(dt, A, B, C):
    """The same scan as one causal matrix per head, as the diagram draws it:

        M[t, s] = (C_t · B_s) Δ_s exp(A Σ_{s<r≤t} Δ_r)    for s ≤ t

    so y_t = Σ_s M[t, s] x_s + D x_t. C plays the query, B the key and x the value.
    Returns (batch, heads, seq, seq)."""
    t = dt.shape[1]
    log_decay = torch.cumsum(dt * A, dim=1)  # (batch, seq, heads)
    gap = log_decay[:, :, None] - log_decay[:, None, :]  # (batch, t, s, heads): A Σ_{s<r≤t} Δ_r
    causal = torch.ones(t, t, dtype=torch.bool, device=dt.device).tril()[..., None]
    decay = torch.exp(gap.masked_fill(~causal, float("-inf")))
    M = (C @ B.transpose(1, 2))[..., None] * dt[:, None] * decay  # (batch, t, s, heads)
    return M.permute(0, 3, 1, 2)


# %% [gated-rmsnorm] Gate · RMSNorm
class GatedRMSNorm(nn.Module):
    """y ⊙ SiLU(z), then RMSNorm. The gate z comes straight from the mixer's input projection."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, y, z):
        y = y * F.silu(z)
        return self.weight * y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [mamba-mixer] Mamba2 mixer
class MambaMixer(nn.Module):
    """in_proj → z | x B C | Δ, a causal conv over x B C, the selective scan, then gate, norm and Wout."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        conv_dim = c.d_inner + 2 * c.d_state  # x, B and C go through the conv together
        self.in_proj = nn.Linear(c.d_model, c.d_inner + conv_dim + c.n_mamba_heads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, c.d_conv, groups=conv_dim)  # applied by causal_conv1d
        self.dt_bias = nn.Parameter(torch.zeros(c.n_mamba_heads))
        self.A_log = nn.Parameter(torch.zeros(c.n_mamba_heads))  # A = −exp(A_log): one decay rate per head
        self.D = nn.Parameter(torch.ones(c.n_mamba_heads))  # skip: each head adds D × its own input
        self.norm = GatedRMSNorm(c.d_inner, c.eps)
        self.out_proj = nn.Linear(c.d_inner, c.d_model, bias=False)

    def forward(self, x):  # x: (batch, seq, d_model)
        c, (b, t, _) = self.c, x.shape
        z, xBC, dt = self.in_proj(x).split([c.d_inner, c.d_inner + 2 * c.d_state, c.n_mamba_heads], dim=-1)
        xBC = causal_conv1d(xBC, self.conv1d.weight, self.conv1d.bias)
        xs, B, C = xBC.split([c.d_inner, c.d_state, c.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias).clamp(min=c.dt_min)  # Δ: one step size per token and head
        y = selective_scan(xs.view(b, t, c.n_mamba_heads, -1), dt, -torch.exp(self.A_log), B, C, self.D)
        return self.out_proj(self.norm(y.reshape(b, t, -1), z))


# %% [block] Mamba and hybrid blocks
class MambaBlock(nn.Module):
    """x + Mixer(RMSNorm(x)). In a hybrid block the shared transformer's output joins the mixer's
    input, but not the residual."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.mixer = MambaMixer(c)

    def forward(self, x, transformer_out=None):
        h = x if transformer_out is None else x + transformer_out
        return x + self.mixer(self.norm(h))


class HybridBlock(nn.Module):
    """Half 1: the shared transformer and this block's own linear layer. Half 2: a Mamba block."""

    def __init__(self, c: Config, shared: SharedTransformer, adapter: int):
        super().__init__()
        self.shared = shared  # the same module object in every hybrid block
        self.adapter = adapter  # which of the six adapters this block uses
        self.linear = nn.Linear(c.d_model, c.d_model, bias=False)
        self.mamba = MambaBlock(c)

    def forward(self, x, embedding):
        return self.mamba(x, self.linear(self.shared(x, embedding, self.adapter)))


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm, then a projection back to one logit per vocabulary token."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding

    def forward(self, x):
        return self.lm_head(self.norm(x))  # logits: (batch, seq, vocab)


def next_token_probabilities(logits, temperature=1.0, top_k=None, top_p=None):
    """The last position's logits, scaled by temperature and cut to top-k or top-p, as probabilities."""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        p = sorted_logits.softmax(dim=-1)
        drop = p.cumsum(dim=-1) - p >= top_p  # keep the smallest set whose mass reaches top_p
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits.softmax(dim=-1)


# %% [model] Full model
class Zamba2(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        shared = SharedTransformer(c)
        self.blocks = nn.ModuleList(
            HybridBlock(c, shared, c.hybrid_layers.index(i)) if i in c.hybrid_layers else MambaBlock(c)
            for i in range(c.n_layers)
        )
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, ids):
        x = embedding = self.embedding(ids)
        for block in self.blocks:
            x = block(x, embedding) if isinstance(block, HybridBlock) else block(x)
        return self.output(x)


# probabilities = next_token_probabilities(Zamba2()(ids), temperature=0.8, top_k=5)
`;export{e as default};
