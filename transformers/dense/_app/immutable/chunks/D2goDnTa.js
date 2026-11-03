const e=`# GCN on Cora (Kipf & Welling 2017), as the GCN tab trains it: two graph convolutions, 1,433 word features → 16 → 7
# topics, each one Â X W with Â = D^-1/2 (A + I) D^-1/2. scripts/graphs/record_gcn.py trains with these blocks; the
# check compares them with torch_geometric's gcn_norm and GCNConv, sklearn's PCA, and the tab's recorded run.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    features: int = 1433  # one entry per word, each paper's row scaled to sum to 1
    hidden: int = 16
    classes: int = 7
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4  # on the first layer only, as in the paper
    epochs: int = 200


# %% [normalize] The normalized adjacency matrix
def gcn_norm(edge_index, nodes):
    """Â's nonzero entries as an edge list: one self-loop per node added, then edge j → i weighted 1 / √(d_i · d_j),
    where d counts the self-loop. Row i of Â holds node i's weights for itself and its neighbours."""
    loops = torch.arange(nodes)
    edge_index = torch.cat([edge_index, torch.stack([loops, loops])], 1)
    degree = torch.zeros(nodes).index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    weight = degree[edge_index[0]].rsqrt() * degree[edge_index[1]].rsqrt()
    return edge_index, weight


# %% [aggregate] One node's sum over its neighbours
def aggregate(messages, edge_index, weight, nodes):
    """out[i] = Σ_j Â_ij · messages[j]: each edge j → i adds its sender's message, scaled by the edge's weight, to the
    receiver. With messages = X W this is the whole product Â (X W), one edge at a time."""
    src, dst = edge_index
    out = torch.zeros(nodes, messages.shape[1], dtype=messages.dtype)
    return out.index_add_(0, dst, weight[:, None] * messages[src])


# %% [gcn-layer] One graph convolution
class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index, weight):
        messages = x @ self.weight.T  # (nodes, d_out): each node's vector after the shared weight
        return aggregate(messages, edge_index, weight, x.shape[0]) + self.bias


# %% [model] Two layers
class GCN(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.conv1 = GCNLayer(c.features, c.hidden)
        self.conv2 = GCNLayer(c.hidden, c.classes)
        self.dropout = c.dropout

    def forward(self, x, edge_index, weight):
        """Returns the logits (nodes, 7) and the hidden layer (nodes, 16), the embedding the tab projects."""
        x = F.dropout(x, self.dropout, self.training)
        hidden = F.relu(self.conv1(x, edge_index, weight))
        logits = self.conv2(F.dropout(hidden, self.dropout, self.training), edge_index, weight)
        return logits, hidden


# %% [train] Training on 140 labelled papers
def train_step(model, optimizer, data, edge_index, weight):
    """One full-batch step: every node's logits are computed, but only the 140 training papers' labels enter the loss."""
    model.train()
    logits, _ = model(data['x'], edge_index, weight)
    loss = F.cross_entropy(logits[data['train_mask']], data['y'][data['train_mask']])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()


def optimizer_for(model, c: Config):
    return torch.optim.Adam([
        {'params': model.conv1.parameters(), 'weight_decay': c.weight_decay},
        {'params': model.conv2.parameters(), 'weight_decay': 0.0},
    ], lr=c.lr)


# %% [projection] The 2-D view of the embeddings
def pca_basis(h):
    """The mean and the top two principal directions of the rows of h (nodes, 16), each direction's sign fixed so its
    largest entry is positive. The tab fits this once, on the trained embeddings, and projects every epoch onto it."""
    mean = h.mean(0)
    _, _, vh = torch.linalg.svd(h - mean, full_matrices=False)
    basis = vh[:2]
    flip = torch.sign(basis.gather(1, basis.abs().argmax(1, keepdim=True)))
    return mean, basis * flip


def project(h, mean, basis):
    return (h - mean) @ basis.T  # (nodes, 2)
`;export{e as default};
