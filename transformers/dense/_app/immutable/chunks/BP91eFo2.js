const e=`# Mamba-2's SSD scan as in mamba2-130m, the model in the Attention Explainer's Mamba-2 tab.
# The mixer replaces attention: 24 heads each scan their own 64 channels, with one scalar decay
# rate per head, so each head's scan is a causal matrix L ∘ (C Bᵀ), the "state space duality".
# Each "# %% [id]" section is one step of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 768
    d_inner: int = 1536  # 2 × d_model
    n_heads: int = 24  # 1536 / 24 = 64 channels per head
    d_state: int = 128  # B and C come in one group, shared by every head
    d_conv: int = 4
    chunk_size: int = 256
    eps: float = 1e-5


# %% [selection] Selection: z, x, B, C, Δ
class Selection(nn.Module):
    """in_proj gives the gate z, then x, B and C, then one Δ per head. A causal depthwise conv and
    SiLU mix every channel of x, B and C with its own last 3 tokens."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        conv_dim = c.d_inner + 2 * c.d_state
        self.in_proj = nn.Linear(c.d_model, c.d_inner + conv_dim + c.n_heads, bias=False)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, c.d_conv, groups=conv_dim, padding=c.d_conv - 1)
        self.dt_bias = nn.Parameter(torch.zeros(c.n_heads))

    def forward(self, u):  # u: (batch, seq, d_model), the input after RMSNorm
        c, (b, t, _) = self.c, u.shape
        z, xBC, dt = self.in_proj(u).split([c.d_inner, c.d_inner + 2 * c.d_state, c.n_heads], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :t].transpose(1, 2))  # keep the causal outputs
        x, B, C = xBC.split([c.d_inner, c.d_state, c.d_state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # Δ > 0: (batch, seq, heads)
        return x.view(b, t, c.n_heads, -1), z, dt, B, C  # x: (batch, seq, heads, 64); B, C: (batch, seq, 128)


# %% [selective-scan] Selective scan
def selective_scan(x, dt, A, B, C, D):
    """The recurrence, one token at a time. Head h keeps a (64 × 128) state, decayed by one scalar:

        S_t = exp(Δ_t A_h) S_{t-1} + Δ_t x_t B_tᵀ,    y_t = S_t C_t + D_h x_t

    x: (batch, seq, heads, 64); dt: (batch, seq, heads); A, D: (heads,); B, C: (batch, seq, 128)."""
    b, t, h, p = x.shape
    S = x.new_zeros(b, h, p, B.shape[-1])
    ys = []
    for i in range(t):
        decay = torch.exp(dt[:, i] * A)[..., None, None]  # how much of each head's state survives
        S = decay * S + (dt[:, i, :, None] * x[:, i])[..., None] * B[:, i, None, None, :]
        ys.append((S @ C[:, i, None, :, None]).squeeze(-1) + D[:, None] * x[:, i])
    return torch.stack(ys, dim=1)  # (batch, seq, heads, 64)


# %% [ssd] SSD: the scan as a matrix
def ssd_matrix(dt, A, B, C):
    """The same scan as one causal matrix per head, as the head card draws it:

        L_h[t, s] = exp(A_h Σ_{s<r≤t} Δ_r[h]),    M_h = L_h ∘ (C Bᵀ) · diag(Δ_h)    for s ≤ t

    so y_t = Σ_s M_h[t, s] x_s + D_h x_t: C plays the query, B the key and x the value, and L is
    the decay mask. Returns M and L, each (batch, heads, seq, seq)."""
    t = dt.shape[1]
    log_decay = torch.cumsum(dt * A, dim=1)  # A Σ_{r≤t} Δ_r: (batch, seq, heads)
    gap = log_decay[:, :, None] - log_decay[:, None, :]  # (batch, t, s, heads)
    causal = torch.ones(t, t, dtype=torch.bool, device=dt.device).tril()[..., None]
    L = torch.exp(gap.masked_fill(~causal, float("-inf"))).permute(0, 3, 1, 2)
    M = L * (C @ B.transpose(1, 2))[:, None] * dt.transpose(1, 2)[:, :, None, :]  # Δ_s scales column s
    return M, L


# %% [chunked-scan] Chunked scan
def chunked_scan(x, dt, A, B, C, D, chunk_size):
    """What the model runs: the matrix form inside each chunk of chunk_size tokens, plus each
    head's (64 × 128) state carried from one chunk to the next. A prompt shorter than one chunk is
    exactly ssd_matrix."""
    b, t, h, p = x.shape
    S = x.new_zeros(b, h, p, B.shape[-1])  # the state entering the chunk
    ys = []
    for start in range(0, t, chunk_size):
        part = slice(start, start + chunk_size)
        xc, dtc, Bc, Cc = x[:, part], dt[:, part], B[:, part], C[:, part]
        M, _ = ssd_matrix(dtc, A, Bc, Cc)
        log_decay = torch.cumsum(dtc * A, dim=1)  # (batch, chunk, heads)
        y = torch.einsum("bhts,bshp->bthp", M, xc)  # tokens of this chunk
        y = y + torch.einsum("bhpn,btn->bthp", S, Cc) * torch.exp(log_decay)[..., None]  # earlier chunks
        ys.append(y + D[:, None] * xc)
        to_end = torch.exp(log_decay[:, -1:] - log_decay) * dtc  # Δ_s exp(A Σ_{s<r≤end} Δ_r)
        S = torch.exp(log_decay[:, -1])[..., None, None] * S + torch.einsum("bsh,bshp,bsn->bhpn", to_end, xc, Bc)
    return torch.cat(ys, dim=1)  # (batch, seq, heads, 64)


# %% [output] Gate, RMSNorm and output projection
class Output(nn.Module):
    """y ⊙ SiLU(z), then RMSNorm over all 1536 channels, then back to d_model."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(c.d_inner))
        self.eps = c.eps
        self.out_proj = nn.Linear(c.d_inner, c.d_model, bias=False)

    def forward(self, y, z):  # y, z: (batch, seq, d_inner)
        y = y * F.silu(z)
        y = self.norm_weight * y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.out_proj(y)


# %% [model] Mamba-2 mixer
class Mamba2Mixer(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c = c
        self.selection = Selection(c)
        self.A_log = nn.Parameter(torch.zeros(c.n_heads))  # A = −exp(A_log): one decay rate per head
        self.D = nn.Parameter(torch.ones(c.n_heads))  # skip: each head adds D × its own x
        self.output = Output(c)

    def forward(self, u):
        b, t, _ = u.shape
        x, z, dt, B, C = self.selection(u)
        y = chunked_scan(x, dt, -torch.exp(self.A_log), B, C, self.D, self.c.chunk_size)
        return self.output(y.reshape(b, t, -1), z)
`;export{e as default};
