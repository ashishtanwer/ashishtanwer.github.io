const e=`# GAT on Cora (Veličković et al. 2018), as the GAT tab trains it: 8 attention heads of 8 features each, concatenated,
# then one head giving the 7 topic scores. Every edge j → i (self-loops included) gets a weight α_ij learned from the two
# nodes' vectors, where GCN uses the fixed 1 / √(d_i · d_j). scripts/graphs/record_gat.py trains with these blocks; the
# check compares them with torch_geometric's GATConv (and its attention weights) and with the tab's recorded run.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    features: int = 1433  # one entry per word, each paper's row scaled to sum to 1
    heads: int = 8
    hidden: int = 8  # per head; the first layer's output is 8 × 8 = 64 wide
    classes: int = 7
    out_heads: int = 1
    dropout: float = 0.6  # on each layer's input and on the attention weights
    slope: float = 0.2  # LeakyReLU's negative slope
    lr: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 200


def with_self_loops(edge_index, nodes):
    loops = torch.arange(nodes)
    return torch.cat([edge_index, torch.stack([loops, loops])], 1)


# %% [attention] Attention over each node's neighbours
def edge_softmax(scores, dst, nodes):
    """Softmax of the scores (edges, heads) over the edges that arrive at the same node, so every node's weights over
    its neighbours sum to 1 per head."""
    peak = torch.full((nodes, scores.shape[1]), -torch.inf).index_reduce_(0, dst, scores, 'amax')
    e = (scores - peak[dst]).exp()
    total = torch.zeros(nodes, scores.shape[1]).index_add_(0, dst, e)
    return e / total[dst]


def attention(z, att_src, att_dst, edge_index, slope):
    """α_ij = softmax_j LeakyReLU(a_srcᵀ z_j + a_dstᵀ z_i) per head. z is (nodes, heads, features), each node's vector
    after the layer's weight; a_src and a_dst are (heads, features). Returns α as (edges, heads)."""
    src, dst = edge_index
    score_src = (z * att_src).sum(-1)  # (nodes, heads): what each node contributes as a sender
    score_dst = (z * att_dst).sum(-1)  # (nodes, heads): ... and as a receiver
    scores = F.leaky_relu(score_src[src] + score_dst[dst], slope)
    return edge_softmax(scores, dst, z.shape[0])


# %% [gat-layer] One multi-head attention layer
class GATLayer(nn.Module):
    def __init__(self, d_in, d_out, heads, concat, c: Config):
        super().__init__()
        self.heads, self.d_out, self.concat, self.c = heads, d_out, concat, c
        self.weight = nn.Parameter(torch.empty(heads * d_out, d_in))
        self.att_src = nn.Parameter(torch.empty(heads, d_out))
        self.att_dst = nn.Parameter(torch.empty(heads, d_out))
        self.bias = nn.Parameter(torch.zeros(heads * d_out if concat else d_out))
        for p in (self.weight, self.att_src, self.att_dst):
            nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index):
        """edge_index must already hold the self-loops. Returns the output and α (edges, heads)."""
        z = (x @ self.weight.T).view(-1, self.heads, self.d_out)
        alpha = attention(z, self.att_src, self.att_dst, edge_index, self.c.slope)
        dropped = F.dropout(alpha, self.c.dropout, self.training)
        src, dst = edge_index
        out = torch.zeros_like(z).index_add_(0, dst, dropped[:, :, None] * z[src])  # Σ_j α_ij z_j per head
        out = out.flatten(1) if self.concat else out.mean(1)
        return out + self.bias, alpha


# %% [model] Two layers
class GAT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.conv1 = GATLayer(c.features, c.hidden, c.heads, True, c)
        self.conv2 = GATLayer(c.heads * c.hidden, c.classes, c.out_heads, False, c)
        self.dropout = c.dropout

    def forward(self, x, edge_index):
        """Returns the logits and both layers' attention weights, (edges, 8) and (edges, 1)."""
        x = F.dropout(x, self.dropout, self.training)
        hidden, alpha1 = self.conv1(x, edge_index)
        hidden = F.elu(hidden)
        logits, alpha2 = self.conv2(F.dropout(hidden, self.dropout, self.training), edge_index)
        return logits, alpha1, alpha2


# %% [entropy] How spread out each node's attention is
def attention_entropy(alpha, dst, nodes):
    """H_i = −Σ_j α_ij log α_ij per node and head (nats). A node with d neighbours (self-loop included) has at most
    log d, when its attention is uniform; GCN's fixed weights are close to uniform when the neighbours' degrees are
    alike."""
    return torch.zeros(nodes, alpha.shape[1]).index_add_(0, dst, -alpha * alpha.clamp_min(1e-30).log())


def head_disagreement(alpha, dst, nodes):
    """Per node, the mean over pairs of heads of the total-variation distance ½ Σ_j |α^h_ij − α^g_ij|: 0 when every
    head weighs the neighbours alike, 1 when two heads put their weight on different neighbours."""
    heads = alpha.shape[1]
    total = torch.zeros(nodes)
    for h in range(heads):
        for g in range(h + 1, heads):
            total.index_add_(0, dst, 0.5 * (alpha[:, h] - alpha[:, g]).abs())
    return total / (heads * (heads - 1) / 2)


# %% [train] Training on 140 labelled papers
def train_step(model, optimizer, data, edge_index):
    """One full-batch step with dropout on; only the 140 training papers' labels enter the loss."""
    model.train()
    logits, _, _ = model(data['x'], edge_index)
    loss = F.cross_entropy(logits[data['train_mask']], data['y'][data['train_mask']])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.detach()
`;export{e as default};
