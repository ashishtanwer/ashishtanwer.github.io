const n=`# MLP-Mixer's token-mixing MLP as in MLP-Mixer B/16, the model in the Attention Explainer's
# MLP-Mixer tab. Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dim: int = 768  # channels
    tokens: int = 196  # 14 × 14 patches of 16 × 16 pixels at 224 × 224
    token_hidden: int = 384  # 0.5 × dim
    channel_hidden: int = 3072  # 4 × dim


# %% [norm] LayerNorm over each token's channels
def norm(dim):
    return nn.LayerNorm(dim, eps=1e-6)


# %% [token-mlp] Token-mixing MLP
class TokenMlp(nn.Module):
    """One MLP mixes the tokens of each channel, and all channels share it. Its first layer has one
    fixed weight per (hidden unit, token)."""

    def __init__(self, tokens, hidden):
        super().__init__()
        self.fc1 = nn.Linear(tokens, hidden)  # W1: (hidden, tokens)
        self.fc2 = nn.Linear(hidden, tokens)  # W2: (tokens, hidden)

    def forward(self, x):  # x: (batch, tokens, channels)
        x = x.transpose(1, 2)  # each channel's tokens become one row
        return self.fc2(F.gelu(self.fc1(x))).transpose(1, 2)


# %% [matrix] The MLP as one T × T matrix per channel, for this input
def effective_mixing(mlp, x):
    """Exact GELU is h · Φ(h), with Φ the standard normal CDF, so for one input
    y_c = M_c x_c + β_c with M_c = W2 diag(Φ(h_c)) W1 and h_c = W1 x_c + b1.
    The gates Φ(h_c) come from x_c: M_c changes with the image and from channel to channel."""
    h = mlp.fc1(x.transpose(1, 2))  # (batch, channels, hidden)
    gate = 0.5 * (1 + torch.erf(h / math.sqrt(2)))
    matrices = torch.einsum("tk,bck,ks->bcts", mlp.fc2.weight, gate, mlp.fc1.weight)
    bias = (gate * mlp.fc1.bias) @ mlp.fc2.weight.T + mlp.fc2.bias
    return matrices, bias  # (batch, channels, tokens, tokens), (batch, channels, tokens)


def mix_with_matrices(x, matrices, bias):
    """Each channel's own matrix mixes that channel's tokens, then its bias is added."""
    out = torch.einsum("bcts,bcs->bct", matrices, x.transpose(1, 2)) + bias
    return out.transpose(1, 2)


# %% [block] Mixer block
class MixerBlock(nn.Module):
    """Token mixing where attention would be, then a per-token channel MLP; both are residual."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.norm1 = norm(c.dim)
        self.token_mlp = TokenMlp(c.tokens, c.token_hidden)
        self.norm2 = norm(c.dim)
        self.fc1 = nn.Linear(c.dim, c.channel_hidden)
        self.fc2 = nn.Linear(c.channel_hidden, c.dim)

    def forward(self, x):  # x: (batch, tokens, channels)
        x = x + self.token_mlp(self.norm1(x))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
`;export{n as default};
