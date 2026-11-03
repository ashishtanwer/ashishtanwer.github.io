const e=`# Message passing on Cora (2,708 papers, 5,278 citations, 1,433 word features, 7 topics), as the Message passing tab
# trains it: GCN layers stacked 1 to 32 deep with no residuals, so every layer mixes each paper with its neighbours once
# more. scripts/graphs/record_message_passing.py trains with these blocks; the check compares them with
# torch_geometric's GCNConv, k_hop_subgraph and dense-matrix versions of each measure.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    features: int = 1433  # one 0/1 entry per word of the vocabulary
    classes: int = 7
    hidden: int = 64
    layers: int = 2  # the tab trains 1, 2, 3, 4, 6, 8, 12, 16, 24 and 32
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 5e-4  # on every layer
    epochs: int = 200


# %% [propagate] One round of message passing
def propagate(x, edge_index, weight, nodes):
    """out[i] = Σ over edges j → i of weight · x[j]: every node sends its vector along its edges and every node adds up
    what arrives. edge_index is (2, edges): row 0 the sender j, row 1 the receiver i."""
    src, dst = edge_index
    out = torch.zeros(nodes, x.shape[1], dtype=x.dtype)
    return out.index_add_(0, dst, weight[:, None] * x[src])


# %% [normalize] Self-loops and degree normalization
def gcn_norm(edge_index, nodes):
    """Add one self-loop per node, then weight the edge j → i by 1 / √(d_i · d_j), d counting the self-loop: the
    entries of D^-1/2 (A + I) D^-1/2."""
    loops = torch.arange(nodes)
    edge_index = torch.cat([edge_index, torch.stack([loops, loops])], 1)
    degree = torch.zeros(nodes).index_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
    weight = degree[edge_index[0]].rsqrt() * degree[edge_index[1]].rsqrt()
    return edge_index, weight


# %% [gcn-layer] One GCN layer
class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index, weight):
        """Transform every node's vector, then pass it to the neighbours: Â (X Wᵀ) + b."""
        return propagate(x @ self.weight.T, edge_index, weight, x.shape[0]) + self.bias


# %% [deep-gcn] L layers deep
class DeepGCN(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        sizes = [c.features] + [c.hidden] * (c.layers - 1) + [c.classes]
        self.layers = nn.ModuleList(GCNLayer(a, b) for a, b in zip(sizes, sizes[1:]))
        self.dropout = c.dropout

    def forward(self, x, edge_index, weight):
        """Returns the logits and every layer's output (after ReLU, except the last layer's logits)."""
        states = []
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, self.dropout, self.training)
            x = layer(x, edge_index, weight)
            if i < len(self.layers) - 1:
                x = F.relu(x)
            states.append(x)
        return x, states


# %% [receptive-field] The nodes that reach a node after k layers
def hops_from(node, edge_index, nodes, hops):
    """distance[j] = the fewest edges from j to \`node\` (−1 if more than \`hops\`). After k layers node's output depends
    on exactly the nodes with distance ≤ k."""
    distance = torch.full((nodes,), -1)
    distance[node] = 0
    frontier = torch.zeros(nodes, dtype=torch.bool)
    frontier[node] = True
    for k in range(1, hops + 1):
        reached = torch.zeros(nodes, dtype=torch.bool)
        reached[edge_index[0][frontier[edge_index[1]]]] = True  # senders into the frontier
        frontier = reached & (distance < 0)
        distance[frontier] = k
    return distance


# %% [influence] How much each node's features move one node's output
def influence(model, x, edge_index, weight, node, layer):
    """I[j] = Σ |∂h_node / ∂x_j| over every output and input feature, scaled to sum to 1 (Xu et al. 2018): the share of
    node's layer-\`layer\` vector that node j's own words account for. One backward pass per output feature."""
    x = x.clone().requires_grad_(True)
    _, states = model(x, edge_index, weight)
    h = states[layer][node]
    total = torch.zeros(x.shape[0])
    for d in range(h.numel()):
        (grad,) = torch.autograd.grad(h[d], x, retain_graph=d < h.numel() - 1)
        total += grad.abs().sum(1)
    return total / total.sum()


# %% [smoothness] Dirichlet energy and cosine similarity
def dirichlet_energy(h, edge_index):
    """E(H) = (1/N) Σ_i Σ_{j ∈ N(i)} ‖h_i − h_j‖² (Rusch et al. 2023): how far each paper's vector still is from its
    neighbours', over every citation in both directions. It is 0 exactly when all connected papers have one vector."""
    src, dst = edge_index
    return (h[src] - h[dst]).pow(2).sum() / h.shape[0]


def mean_square_norm(h):
    """(1/N) Σ_i ‖h_i‖², the scale E(H) is measured against: energy can also fall because every vector shrinks."""
    return h.pow(2).sum() / h.shape[0]


def mean_cosine(h):
    """The mean cosine similarity over all pairs of distinct nodes, from the sum of unit vectors:
    ‖Σ u_i‖² = N + Σ_{i≠j} u_i · u_j. An all-zero row counts as similarity 0 to everything."""
    norm = h.norm(dim=1, keepdim=True)
    u = torch.where(norm > 0, h / norm.clamp_min(1e-30), torch.zeros_like(h))
    n = h.shape[0]
    alive = (norm > 0).sum()
    return ((u.sum(0).pow(2).sum() - alive) / (n * (n - 1))).item()


# %% [train] Training on the 140 labelled papers
def train(model, data, c: Config):
    """Adam on the cross-entropy of the 140 training papers, the whole graph in every step (full-batch). Returns the loss
    and the train / validation / test accuracy after every epoch."""
    edge_index, weight = gcn_norm(data['edge_index'], data['x'].shape[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    history = []
    for _ in range(c.epochs):
        model.train()
        logits, _ = model(data['x'], edge_index, weight)
        loss = F.cross_entropy(logits[data['train_mask']], data['y'][data['train_mask']])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append([loss.item(), *accuracy(model, data, edge_index, weight)])
    return history


@torch.no_grad()
def accuracy(model, data, edge_index, weight):
    model.eval()
    predicted = model(data['x'], edge_index, weight)[0].argmax(1)
    return [(predicted[m] == data['y'][m]).float().mean().item() for m in
            (data['train_mask'], data['val_mask'], data['test_mask'])]
`;export{e as default};
