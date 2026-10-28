const n=`# Mamba's selective scan as in mamba-130m, the model in the Attention Explainer's Mamba tab.
# The mixer replaces attention: each of its 1536 channels runs its own scan over the tokens.
# Each "# %% [id]" section is one step of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_model: int = 768
    d_inner: int = 1536  # 2 × d_model channels, each scanned on its own
    d_state: int = 16  # each channel keeps 16 state entries, each with its own decay rate
    dt_rank: int = 48  # Δ is predicted through this bottleneck: ⌈768 / 16⌉
    d_conv: int = 4


# %% [selection] Selection: x, z, Δ, B, C
class Selection(nn.Module):
    """in_proj splits each token into the scan's input x and a gate z. A causal depthwise conv mixes
    each channel of x with its own last 3 tokens. x_proj then selects, per token, the step size Δ
    and the vectors B (what to write into the state) and C (what to read out of it)."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.in_proj = nn.Linear(c.d_model, 2 * c.d_inner, bias=False)
        self.conv1d = nn.Conv1d(c.d_inner, c.d_inner, c.d_conv, groups=c.d_inner, padding=c.d_conv - 1)
        self.x_proj = nn.Linear(c.d_inner, c.dt_rank + 2 * c.d_state, bias=False)
        self.dt_proj = nn.Linear(c.dt_rank, c.d_inner)

    def forward(self, u):  # u: (batch, seq, d_model), the input after RMSNorm
        t = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)  # (batch, seq, d_inner) each
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :t].transpose(1, 2))  # keep the causal outputs
        dt, B, C = self.x_proj(x).split([self.c.dt_rank, self.c.d_state, self.c.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # Δ > 0: (batch, seq, d_inner), one step per token and channel
        return x, z, dt, B, C  # B, C: (batch, seq, d_state), shared by every channel


# %% [selective-scan] Selective scan
def selective_scan(x, dt, A, B, C, D):
    """The recurrence, one token at a time. Channel c keeps a 16-entry state h:

        h_t = exp(Δ_t A_c) ⊙ h_{t-1} + Δ_t B_t x_t,    y_t = C_t · h_t + D_c x_t

    x, dt: (batch, seq, d_inner); A: (d_inner, d_state), all negative; B, C: (batch, seq, d_state)."""
    b, t, d = x.shape
    h = x.new_zeros(b, d, A.shape[1])
    ys = []
    for i in range(t):
        decay = torch.exp(dt[:, i, :, None] * A)  # how much of each state entry survives this step
        h = decay * h + (dt[:, i] * x[:, i])[..., None] * B[:, i, None, :]
        ys.append((h @ C[:, i, :, None]).squeeze(-1) + D * x[:, i])
    return torch.stack(ys, dim=1)  # (batch, seq, d_inner)


# %% [mixing-matrix] The scan as a matrix
def mixing_matrix(dt, A, B, C):
    """The same scan as one causal T × T matrix per channel, as the head card draws it:

        M_c[t, s] = Σ_n C_t[n] · exp(A[c, n] Σ_{s<r≤t} Δ_r[c]) · Δ_s[c] B_s[n]    for s ≤ t

    so y_t[c] = Σ_s M_c[t, s] x_s[c] + D_c x_t[c]: C plays the query, Δ·B the key, x the value.
    The decay sits inside the sum over n, so M_c is not one decay mask times C·Bᵀ.
    Returns (batch, d_inner, seq, seq)."""
    t = dt.shape[1]
    total = dt.cumsum(dim=1)  # Σ_{r≤t} Δ_r: (batch, seq, d_inner)
    gap = total[:, :, None] - total[:, None, :]  # (batch, t, s, d_inner): Σ_{s<r≤t} Δ_r
    causal = torch.ones(t, t, dtype=torch.bool, device=dt.device).tril()
    gap = gap.masked_fill(~causal[..., None], 0.0)  # s > t is zeroed below; 0 keeps exp finite
    decay = torch.exp(gap[..., None] * A)  # (batch, t, s, d_inner, d_state)
    M = torch.einsum("btn,bsn,btscn->bcts", C, B, decay) * dt.transpose(1, 2)[:, :, None, :]
    return M.masked_fill(~causal, 0.0)


# %% [output] Gate and output projection
class Output(nn.Module):
    """y ⊙ SiLU(z), then back to d_model. The gate z skipped the conv and the scan."""

    def __init__(self, c: Config):
        super().__init__()
        self.out_proj = nn.Linear(c.d_inner, c.d_model, bias=False)

    def forward(self, y, z):
        return self.out_proj(y * F.silu(z))  # (batch, seq, d_model)


# %% [model] Mamba mixer
class MambaMixer(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.selection = Selection(c)
        self.A_log = nn.Parameter(torch.zeros(c.d_inner, c.d_state))  # A = −exp(A_log): decay rates
        self.D = nn.Parameter(torch.ones(c.d_inner))  # skip: each channel adds D × its own x
        self.output = Output(c)

    def forward(self, u):
        x, z, dt, B, C = self.selection(u)
        y = selective_scan(x, dt, -torch.exp(self.A_log), B, C, self.D)
        return self.output(y, z)
`;export{n as default};
