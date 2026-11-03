const e=`# μP, the maximal update parametrization (Yang et al. 2021), for the Pretraining Explainer's small GPT (4 layers,
# 4 heads, context 256, an LM head tied to the token embedding, AdamW on TinyStories), trained at widths 64 … 512.
# Under the standard parametrization (SP: every weight drawn with std 0.02, one learning rate for all of them) the
# best learning rate shrinks as the model widens; μP rescales each kind of weight by the width multiplier
# m = d_model / base_width (its initial std, its learning rate, and the attention and output multipliers), so the
# learning rate tuned at one width is the right one at every width. At d_model = base_width every multiplier is 1
# and μP is SP. scripts/pretraining/record_mup.py trains the tab's sweep with this model.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    base_width: int = 256  # the shared setup's d_model: tune here, transfer everywhere
    base_head_dim: int = 64  # its head width (4 heads)
    init_std: float = 0.02  # GPT-2's std, the SP model's for every matrix
    vocab_size: int = 50257
    context: int = 256
    n_layer: int = 4
    n_head: int = 4  # fixed, so the heads widen with the model


# %% [width-mult] The width multiplier
def width_mult(d_model, c: Config):
    """m = d_model / base_width: 0.25 at width 64, 1 at the base width 256, 2 at 512."""
    return d_model / c.base_width


# %% [init] Initialisation
def init_std(name, d_model, c: Config):
    """Input weights (the token and position embeddings) keep their std: each output coordinate reads one row, whatever
    the width. Hidden matrices sum over d_model inputs, so their std shrinks as 1/√m to keep activations O(1); the two
    projections back into the residual stream keep GPT-2's extra 1/√(2·n_layer)."""
    m = width_mult(d_model, c)
    if name in ("wte.weight", "wpe.weight"):
        return c.init_std
    if name.endswith(("attn.o.weight", "mlp.proj.weight")):
        return c.init_std / math.sqrt(2 * c.n_layer) / math.sqrt(m)
    return c.init_std / math.sqrt(m)


# %% [attention] Attention scaled by 1/d_head
class Attention(nn.Module):
    """q·k grows like d_head once q and k are correlated (as training makes them), not like √d_head, so μP divides the
    scores by d_head; written as (√base_head_dim / d_head) so that it is SP's 1/√d_head at the base width."""

    def __init__(self, d_model, c: Config):
        super().__init__()
        self.n_head, self.head_dim = c.n_head, d_model // c.n_head
        self.scale = math.sqrt(c.base_head_dim) / self.head_dim
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) for proj in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model, bias=False)
        self.proj = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, d_model, c: Config):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(d_model), Attention(d_model, c)
        self.ln2, self.mlp = nn.LayerNorm(d_model), MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


# %% [readout] The output multiplier
def readout(h, wte, d_model, c: Config):
    """The tied head reads d_model coordinates that each move by O(1) during training, so the logits would grow like
    m; μP multiplies them by 1/m. The embedding's own learning rate is left alone (it is an input weight)."""
    return (h @ wte.T) / width_mult(d_model, c)


# %% [model] The μP GPT
class GPT(nn.Module):
    def __init__(self, d_model, c: Config):
        super().__init__()
        self.d_model, self.c = d_model, c
        self.wte = nn.Embedding(c.vocab_size, d_model)
        self.wpe = nn.Embedding(c.context, d_model)
        self.blocks = nn.ModuleList(Block(d_model, c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return readout(self.ln_f(x), self.wte.weight, self.d_model, self.c)

    @torch.no_grad()
    def init_weights(self, generator):
        for name, p in self.named_parameters():
            if p.ndim == 2:
                p.copy_(torch.randn(p.shape, generator=generator) * init_std(name, self.d_model, self.c))
            else:  # LayerNorm gains 1, biases 0
                nn.init.ones_(p) if name.endswith("weight") else nn.init.zeros_(p)
        return self


# %% [lr] Per-layer learning rates
def lr_mult(name, p, d_model, c: Config):
    """Adam's update has the same size whatever the gradient's scale, so a hidden matrix's update is ~lr per weight and
    moves its d_model-sum output by ~lr·d_model: μP divides the hidden matrices' learning rate by m. Embeddings,
    LayerNorm gains and biases are vectors along the width and keep the base learning rate."""
    hidden = p.ndim == 2 and name not in ("wte.weight", "wpe.weight")
    return 1 / width_mult(d_model, c) if hidden else 1.0


def param_groups(model, weight_decay):
    """AdamW's groups: each weight with its learning-rate multiplier, decay on the matrices and the embeddings."""
    groups = {}
    for name, p in model.named_parameters():
        key = (lr_mult(name, p, model.d_model, model.c), weight_decay if p.ndim == 2 else 0.0)
        groups.setdefault(key, []).append(p)
    return [{"params": ps, "lr_mult": mult, "weight_decay": wd} for (mult, wd), ps in groups.items()]


def build(d_model, c: Config, generator):
    return GPT(d_model, c).init_weights(generator)
`;export{e as default};
