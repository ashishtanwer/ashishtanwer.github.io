const e=`# A graph transformer on ZINC molecules, as the Graph Transformer tab trains it: every atom attends to every other atom
# of its molecule (full attention, as in Dwivedi & Bresson 2020's full-graph variant), and the only thing that tells
# atoms where they sit is a Laplacian positional encoding, the molecule's first 8 nontrivial Laplacian eigenvectors, with
# their signs flipped at random in training. The GCN baseline trained on the same split sees the bonds directly. Neither
# model reads bond types. scripts/graphs/record_graph_transformer.py trains both with these blocks; the check compares
# them with torch_geometric's AddLaplacianEigenvectorPE and GCNConv, nn.MultiheadAttention, nn.TransformerEncoderLayer
# and the tab's recorded run.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    atom_types: int = 28  # ZINC's atom vocabulary: C, O, N, F, C H1, S, Cl, O -, ...
    pe_dim: int = 8  # Laplacian eigenvectors per atom
    d_model: int = 64
    heads: int = 4
    d_ff: int = 128
    layers: int = 4
    gcn_hidden: int = 172  # the GCN baseline's width, chosen for about as many weights as the transformer
    lr: float = 1e-3  # Adam, cosine decay to lr / 100
    batch: int = 128
    epochs: int = 100


# %% [laplacian-pe] Laplacian eigenvectors as positions
def laplacian_pe(adjacency, k):
    """The eigenvectors of L = I − D^-1/2 A D^-1/2 with the k smallest eigenvalues after the first (whose vector only
    reflects the degrees), one row per atom: (atoms, k), zero-padded when the molecule has ≤ k atoms. Each column is
    defined only up to its sign, and up to a rotation when eigenvalues repeat."""
    a = adjacency.double()
    inv_sqrt = a.sum(1).clamp_min(1).rsqrt()
    lap = torch.eye(len(a), dtype=a.dtype) - inv_sqrt[:, None] * a * inv_sqrt[None, :]
    values, vectors = torch.linalg.eigh(lap)  # ascending
    pe = torch.zeros(len(a), k, dtype=torch.float64)
    m = min(k, len(a) - 1)
    pe[:, :m] = vectors[:, 1:m + 1]
    return pe.float(), values.float()


def random_signs(k, generator=None):
    """One ±1 per eigenvector, drawn afresh for every training batch; the whole batch shares it."""
    return torch.where(torch.rand(k, generator=generator) >= 0.5, 1.0, -1.0)


# %% [attention] Full multi-head attention over a molecule's atoms
class FullAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x, mask):
        """x (batch, atoms, d), mask (batch, atoms) True for real atoms. Every atom attends to every real atom of its
        own molecule, bonded or not. Returns the output and the weights (batch, heads, atoms, atoms)."""
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)  # padding is never attended to
        weights = scores.softmax(-1)
        out = (weights @ v).transpose(1, 2).reshape(b, n, d)
        return self.out(out), weights


# %% [block] One transformer layer
class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attention = FullAttention(c)
        self.ff = nn.Sequential(nn.Linear(c.d_model, c.d_ff), nn.ReLU(), nn.Linear(c.d_ff, c.d_model))

    def forward(self, x, mask):
        a, weights = self.attention(self.norm1(x), mask)
        x = x + a
        return x + self.ff(self.norm2(x)), weights


# %% [model] Graph transformer
class GraphTransformer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.atoms = nn.Embedding(c.atom_types, c.d_model)
        self.pe = nn.Linear(c.pe_dim, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Sequential(nn.Linear(c.d_model, c.d_model // 2), nn.ReLU(), nn.Linear(c.d_model // 2, 1))

    def forward(self, atoms, pe, mask):
        """atoms (batch, n) atom types, pe (batch, n, 8), mask (batch, n). Returns one prediction per molecule and every
        layer's attention weights."""
        x = self.atoms(atoms) + self.pe(pe)
        maps = []
        for block in self.blocks:
            x, weights = block(x, mask)
            maps.append(weights)
        x = self.norm(x)
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)  # mean over the real atoms
        return self.head(pooled).squeeze(-1), maps


# %% [gcn-baseline] The GCN baseline
def normalized_adjacency(adjacency, mask):
    """Â = D^-1/2 (A + I) D^-1/2 for a padded batch (batch, n, n); padded atoms get no edges and no self-loop."""
    a = adjacency + torch.diag_embed(mask.float())
    inv_sqrt = a.sum(-1).clamp_min(1).rsqrt()
    return inv_sqrt[..., :, None] * a * inv_sqrt[..., None, :]


class GCNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, a_hat):
        """x + ReLU(LayerNorm(Â X W + b)): a graph convolution with a residual, as in Dwivedi et al.'s GCN baseline."""
        return x + F.relu(self.norm(a_hat @ self.weight(x) + self.bias))


class GCN(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.gcn_hidden
        self.atoms = nn.Embedding(c.atom_types, d)
        self.layers = nn.ModuleList(GCNLayer(d) for _ in range(c.layers))
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    def forward(self, atoms, adjacency, mask):
        x = self.atoms(atoms)
        a_hat = normalized_adjacency(adjacency, mask)
        for layer in self.layers:
            x = layer(x, a_hat)
        pooled = (x * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.head(pooled).squeeze(-1)


# %% [train] Training on 10,000 molecules
def train_step(model, optimizer, batch, signs=None):
    """One step of the mean absolute error. The transformer's batch gets its eigenvectors' signs flipped by \`signs\`,
    so it never sees one fixed orientation; the GCN's batch is used as it is."""
    model.train()
    if signs is not None:
        prediction, _ = model(batch['atoms'], batch['pe'] * signs, batch['mask'])
    else:
        prediction = model(batch['atoms'], batch['adjacency'], batch['mask'])
    loss = (prediction - batch['y']).abs().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()
`;export{e as default};
