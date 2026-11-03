const e=`# Initialization and normalization, as the Pretraining Explainer records them: a deep, narrow GPT (32 blocks, d_model
# 128, 4 heads, context 128, the TinyStories data and GPT-2 tokenizer of the other tabs) built with one of four
# initializations and one of four normalization layouts, then trained for a few hundred AdamW steps. At the start and
# after chosen steps the run measures, at every depth, how large the residual stream is (and how much that varies from
# token to token), how much each attention and MLP block writes into it, how large the loss's gradient with respect to
# it is, and how spread out each layer's attention is. scripts/pretraining/init_norm_runs.py trains with these blocks.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    context: int = 128
    n_layer: int = 32
    n_head: int = 4
    d_model: int = 128
    init: str = "default"  # "default" | "small" | "large" | "scaled"
    norm: str = "pre"  # "none" | "post" | "pre" | "rms"
    eps: float = 1e-5  # inside both norms' square roots


# %% [init] Four initializations
STD = {"default": 0.02, "small": 0.002, "large": 0.2, "scaled": 0.02}


def init_weights(model, c: Config, generator):
    """Every matrix (embeddings included) from N(0, σ²): σ = 0.02 (GPT-2's), 0.002 (too small) or 0.2 (too large).
    "scaled" is GPT-2's full rule: σ = 0.02, but the two projections that write into the residual stream in each block
    (attention's output and the MLP's second layer) get 0.02 / √(2·n_layer), so 2·n_layer of them add up to about
    one. Norm gains start at 1 and biases at 0."""
    std = STD[c.init]
    for name, p in model.named_parameters():
        if p.ndim < 2:
            nn.init.ones_(p) if name.endswith("gain") else nn.init.zeros_(p)
            continue
        s = std / math.sqrt(2 * c.n_layer) if c.init == "scaled" and name.endswith(("attn.o.weight", "mlp.proj.weight")) else std
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=generator) * s)
    return model


# %% [norms] LayerNorm and RMSNorm
class LayerNorm(nn.Module):
    """Subtract each token's mean over its d_model features, divide by their standard deviation, then a learned gain
    and bias per feature."""

    def __init__(self, d, eps):
        super().__init__()
        self.gain, self.bias, self.eps = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d)), eps

    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gain + self.bias


class RMSNorm(nn.Module):
    """Divide each token by the root mean square of its features (no mean subtracted, no bias), then a learned gain."""

    def __init__(self, d, eps):
        super().__init__()
        self.gain, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.gain


def make_norm(c: Config):
    if c.norm == "none":
        return nn.Identity()
    return RMSNorm(c.d_model, c.eps) if c.norm == "rms" else LayerNorm(c.d_model, c.eps)


# %% [attention] Causal attention that reports its spread
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_head = c.n_head
        self.q, self.k, self.v, self.o = (nn.Linear(c.d_model, c.d_model, bias=False) for _ in range(4))
        self.entropy = None  # filled when the model is asked to measure

    def forward(self, x, measure=False):
        B, T, D = x.shape
        q, k, v = (p(x).view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for p in (self.q, self.k, self.v))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.n_head)
        scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1), float("-inf"))
        weights = scores.softmax(-1)  # (B, heads, T, T)
        if measure:  # each query's entropy in nats, averaged over heads and queries; ln(i + 1) for uniform over i + 1 keys
            self.entropy = -(weights * weights.clamp_min(1e-30).log()).sum(-1).mean()
        return self.o((weights @ v).transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.proj = nn.Linear(4 * c.d_model, c.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


# %% [block] Where the norm goes
class Block(nn.Module):
    """none: x + f(x), nothing rescales the stream. post (the original Transformer): norm(x + f(x)), the stream itself
    is normalized after every block. pre (GPT-2 onwards) and rms: x + f(norm(x)), only each block's input is
    normalized and the stream is left to grow; one more norm before the head."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.norm1, self.attn, self.norm2, self.mlp = make_norm(c), Attention(c), make_norm(c), MLP(c)
        self.writes = {}  # RMS of what attention and the MLP add, when measuring

    def forward(self, x, measure=False):
        if self.c.norm == "post":
            a = self.attn(x, measure)
            x = self.norm1(x + a)
            m = self.mlp(x)
            x = self.norm2(x + m)
        else:
            a = self.attn(self.norm1(x), measure)
            x = x + a
            m = self.mlp(self.norm2(x))
            x = x + m
        if measure:
            self.writes = {"attn": a.detach().pow(2).mean().sqrt(), "mlp": m.detach().pow(2).mean().sqrt()}
        return x


# %% [model] The deep GPT
class DeepGPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d_model)
        self.wpe = nn.Embedding(c.context, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm_f = make_norm(c) if c.norm in ("pre", "rms") else nn.Identity()

    def forward(self, idx, measure=False):
        """Logits; with measure=True also the residual stream after the embedding and after every block (n_layer + 1
        tensors of (batch, T, d_model)), kept in the graph so their gradients can be read."""
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        stream = [x]
        for block in self.blocks:
            x = block(x, measure)
            stream.append(x)
        for s in stream if measure else ():
            s.retain_grad()
        logits = self.norm_f(x) @ self.wte.weight.T  # head tied to the embedding
        return (logits, stream) if measure else logits


def build(c: Config, seed: int):
    return init_weights(DeepGPT(c), c, torch.Generator().manual_seed(seed))


# %% [stats] Measuring the stream and its gradient
QUANTILES = (0.0, 0.1, 0.5, 0.9, 1.0)


def spread(t):
    """Each token's root mean square over its d_model features, then the RMS over all tokens and the quantiles of the
    per-token values (min, 10 %, median, 90 %, max): how large the layer is and how unevenly its tokens are."""
    per_token = t.detach().float().pow(2).mean(-1).sqrt().flatten().cpu()
    return torch.cat([per_token.pow(2).mean().sqrt()[None], torch.quantile(per_token, torch.tensor(QUANTILES))])


def measure(model, x, y):
    """One forward and backward pass on a fixed batch. Returns the loss and, per depth, the stream's spread, the spread
    of the gradient of the loss summed over the batch's tokens (so a token's gradient does not shrink with the batch),
    what each block's attention and MLP wrote, each block's attention entropy and the gradient norm of each block's
    weights."""
    model.zero_grad(set_to_none=True)
    logits, stream = model(x, measure=True)
    loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction="sum")
    loss.backward()
    n = y.numel()
    return {
        "loss": float(loss.detach()) / n,
        "stream": torch.stack([spread(s) for s in stream]),
        "grad": torch.stack([spread(s.grad) for s in stream]),
        "attnWrite": torch.stack([b.writes["attn"] for b in model.blocks]),
        "mlpWrite": torch.stack([b.writes["mlp"] for b in model.blocks]),
        "entropy": torch.stack([b.attn.entropy.detach() for b in model.blocks]),
        "weightGrad": torch.stack([torch.cat([p.grad.reshape(-1) for p in b.parameters()]).norm() / n for b in model.blocks]),
    }


# %% [train-step] One training step
def train_step(model, x, y, optimizer):
    """The mean next-token loss of the batch, one AdamW step; a run that turns inf or NaN is stopped there."""
    loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    optimizer.step()
    return loss.detach(), norm
`;export{e as default};
