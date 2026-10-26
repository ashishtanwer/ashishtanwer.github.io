const e=`# ESM-2 t12 35M, the protein language model in the Protein tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 33  # 20 amino acids, rare residues, gap and special tokens
    d_model: int = 480
    n_layers: int = 12
    n_heads: int = 20  # 480 / 20 = 24 dimensions per head
    d_mlp: int = 1920  # 4 × d_model
    rope_theta: float = 10000.0
    mask_ratio_train: float = 0.15 * 0.8  # share of residues replaced by <mask> in training
    mask_id: int = 32
    eps: float = 1e-5


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token lookup with token dropout: <mask> rows are zeroed and the rest rescaled to the training mask rate."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.c = c

    def forward(self, ids):  # ids: (batch, seq), <cls> residues… <eos>
        masked = ids == self.c.mask_id
        x = self.token(ids).masked_fill(masked[..., None], 0.0)
        observed = masked.float().mean(dim=-1)  # share of this sequence that is masked
        scale = (1 - self.c.mask_ratio_train) / (1 - observed)  # 0.88 × 78/77 with one mask in 78 tokens
        return x * scale[:, None, None]  # (batch, seq, d_model); positions enter through RoPE


# %% [rope] RoPE
class Rotary(nn.Module):
    """Rotate each pair (i, i + d/2) of a head vector by position × inv_freq[i]."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        # The checkpoint stores these frequencies rounded (10000^(-2/24) = 0.46416 is kept as 0.46411),
        # so they are a buffer loaded with the weights rather than recomputed.
        self.register_buffer("inv_freq", theta ** (-torch.arange(0, head_dim, 2).float() / head_dim))

    def forward(self, x):  # x: (batch, heads, seq, head_dim)
        angles = torch.arange(x.shape[-2], device=x.device)[:, None] * self.inv_freq  # (seq, head_dim/2)
        cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Multi-head self-attention
class Attention(nn.Module):
    """Bidirectional attention: every residue sees every other. Also returns the weights for the contact head."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.rotary = Rotary(c.d_model // c.n_heads, c.rope_theta)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model), already LayerNormed
        b, t, d = x.shape
        q, k, v = (proj(x).view(b, t, self.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        q = q * q.shape[-1] ** -0.5  # ESM scales the query by 1/sqrt(24) before rotating it
        q, k = self.rotary(q), self.rotary(k)
        weights = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # (batch, heads, seq, seq), no causal mask
        heads = weights @ v  # (batch, heads, seq, head_dim)
        return self.out(heads.transpose(1, 2).reshape(b, t, d)), weights


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)
        self.down = nn.Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))  # exact (erf) GELU


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        attended, weights = self.attention(self.norm1(x))
        x = x + attended
        return x + self.mlp(self.norm2(x)), weights


# %% [output] Masked residue
class Output(nn.Module):
    """Final LayerNorm, then the LM head: dense → GELU → LayerNorm → tied vocabulary projection + bias."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.dense = nn.Linear(c.d_model, c.d_model)
        self.head_norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.decoder = nn.Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding
        self.bias = nn.Parameter(torch.zeros(c.vocab_size))

    def forward(self, x):
        h = self.head_norm(F.gelu(self.dense(self.norm(x))))
        return self.decoder(h) + self.bias  # logits: (batch, seq, 33)


# %% [apc] Symmetrize and APC
def contact_features(attention):
    """Every block's attention maps → one symmetric, background-corrected map per head."""
    a = attention[..., 1:-1, 1:-1]  # (batch, layers, heads, residues, residues), <cls> and <eos> dropped
    b, layers, heads, n, _ = a.shape
    a = a.reshape(b, layers * heads, n, n)  # 12 × 20 = 240 maps
    s = a + a.transpose(-1, -2)  # A + Aᵀ: contact is symmetric
    # Average product correction: subtract each map's row sum × column sum / total, the part
    # explained by residues that every residue attends to.
    background = s.sum(-1, keepdim=True) * s.sum(-2, keepdim=True) / s.sum((-1, -2), keepdim=True)
    return s - background  # (batch, 240, residues, residues)


# %% [contact-head] Contact prediction
class ContactHead(nn.Module):
    """Logistic regression over the 240 maps: p(i, j) = σ(Σ w·F[i, j] + b)."""

    def __init__(self, c: Config):
        super().__init__()
        self.regression = nn.Linear(c.n_layers * c.n_heads, 1)

    def forward(self, attention):  # attention: (batch, layers, heads, seq, seq)
        features = contact_features(attention).permute(0, 2, 3, 1)  # (batch, residues, residues, 240)
        return torch.sigmoid(self.regression(features).squeeze(-1))  # (batch, residues, residues)


# %% [model] Full model
class ESM2(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.decoder.weight = self.embedding.token.weight  # weight tying
        self.contact_head = ContactHead(c)

    def forward(self, ids):
        x = self.embedding(ids)
        maps = []
        for block in self.blocks:
            x, weights = block(x)
            maps.append(weights)
        attention = torch.stack(maps, dim=1)  # (batch, layers, heads, seq, seq)
        return self.output(x), self.contact_head(attention)


# With one residue masked, logits[0, mask_position].softmax(-1) gives the tab's amino-acid bars.
# On the unmasked sequence, the contact map is p(i, j) for every residue pair.
# logits, contacts = ESM2()(ids)
`;export{e as default};
