const e=`# MusicGen small (Copet et al. 2023), facebook/musicgen-small, as plain PyTorch: a 24-layer transformer decoder that
# writes music as EnCodec codes. EnCodec 32 kHz turns every 640 samples (20 ms) into 4 codes, one per residual
# codebook of 2,048 vectors. MusicGen predicts all 4 of a step at once, one head per codebook, but codebook k runs
# k steps behind codebook 1 (the delay pattern), so each code is predicted after the coarser codes of its own frame.
# The prompt enters through cross-attention to T5-base's output. Generation runs two passes per step, with and
# without the prompt, and mixes their logits (classifier-free guidance) before drawing from the top 250.
# Each "# %% [id]" section is one part of the MusicGen tab's diagram; the T5 encoder and EnCodec's convolutional
# decoder are the released models' own.
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    hidden: int = 1024
    layers: int = 24
    heads: int = 16
    ffn: int = 4096
    codebooks: int = 4  # EnCodec 32 kHz codebooks, one head each
    vocab: int = 2048  # vectors per codebook
    pad: int = 2048  # the extra id: the start token and the delay padding
    text_hidden: int = 768  # T5-base's output width
    max_positions: int = 2048
    guidance: float = 3.0  # classifier-free guidance scale
    top_k: int = 250
    sampling_rate: int = 32000
    hop: int = 640  # samples per EnCodec frame: 50 frames a second
    codec_dim: int = 128  # EnCodec's latent width


# %% [delay-pattern] The delay pattern: codebook k runs k steps late
def delay_pattern(frames, c: Config):
    """(codebooks, frames + codebooks): which frame each step writes for each codebook, −1 for padding. Step 0 is
    the start token; at step s, codebook k writes frame s − 1 − k."""
    step = torch.arange(frames + c.codebooks)
    frame = step[None] - 1 - torch.arange(c.codebooks)[:, None]
    return torch.where((frame >= 0) & (frame < frames), frame, -1)


def delay(codes, c: Config):
    """codes (codebooks, frames) → the delayed grid the decoder reads and writes, padding where there is no frame."""
    frame = delay_pattern(codes.shape[1], c)
    grid = codes.gather(1, frame.clamp(min=0))
    return torch.where(frame >= 0, grid, c.pad)


def undelay(grid, c: Config):
    """The delayed grid back to codes (codebooks, frames), each codebook shifted back by its delay."""
    frames = grid.shape[1] - c.codebooks
    return torch.stack([grid[k, 1 + k:1 + k + frames] for k in range(c.codebooks)])


# %% [embedding] A step's 4 codes become one vector
def sinusoids(positions, width):
    """Fixed positions, cosines then sines, at frequencies 1 down to 1/10,000."""
    half = width // 2
    rate = torch.exp(torch.arange(half).float() * -(math.log(10000) / (half - 1)))
    angle = torch.arange(positions).float()[:, None] * rate[None]
    return torch.cat([angle.cos(), angle.sin()], 1)


class Embedding(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(c.vocab + 1, c.hidden) for _ in range(c.codebooks))
        self.register_buffer("positions", sinusoids(c.max_positions, c.hidden), persistent=False)

    def forward(self, grid):
        """grid (batch, codebooks, steps) → (batch, steps, hidden): one table per codebook, summed, plus positions."""
        x = sum(table(grid[:, k]) for k, table in enumerate(self.tables))
        return x + self.positions[: grid.shape[-1]]


# %% [text-conditioning] The prompt: T5's output projected to the decoder's width
class TextConditioning(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.project = nn.Linear(c.text_hidden, c.hidden)

    def forward(self, t5_out, mask):
        """t5_out (batch, words, 768), mask (batch, words) → keys and values for every cross-attention layer.
        The null prompt for guidance is all zeros with an all-zero mask."""
        return self.project(t5_out) * mask[..., None]


# %% [attention] Self-attention over the steps so far
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.q = nn.Linear(c.hidden, c.hidden, bias=False)
        self.k = nn.Linear(c.hidden, c.hidden, bias=False)
        self.v = nn.Linear(c.hidden, c.hidden, bias=False)
        self.out = nn.Linear(c.hidden, c.hidden, bias=False)

    def forward(self, x, context=None, mask=None):
        """x (batch, steps, hidden). Self-attention reads x itself under a causal mask; cross-attention passes the
        prompt as \`context\` and its padding mask."""
        b, n, d = x.shape
        source = x if context is None else context
        split = lambda t: t.view(b, -1, self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.q(x)), split(self.k(source)), split(self.v(source))
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // self.heads)  # (batch, heads, steps, keys)
        if context is None:
            scores = scores + torch.full((n, n), float("-inf")).triu(1)
        else:  # masked words get the most negative float; an all-masked (null) prompt reads zeros evenly
            scores = scores + (1 - mask[:, None, None, :].float()) * torch.finfo(scores.dtype).min
        weights = scores.softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d)), weights


# %% [cross-attention] Cross-attention: each step reads the prompt
class CrossAttention(Attention):
    def forward(self, x, context, mask):
        return super().forward(x, context, mask)


# %% [mlp] The feed-forward layer
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc1 = nn.Linear(c.hidden, c.ffn, bias=False)
        self.fc2 = nn.Linear(c.ffn, c.hidden, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# %% [block] One decoder block: self-attention, cross-attention, MLP, each pre-norm with a residual
class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(c.hidden), nn.LayerNorm(c.hidden), nn.LayerNorm(c.hidden)
        self.attention = Attention(c)
        self.cross = CrossAttention(c)
        self.mlp = MLP(c)

    def forward(self, x, text, mask):
        x = x + self.attention(self.norm1(x))[0]
        x = x + self.cross(self.norm2(x), text, mask)[0]
        return x + self.mlp(self.norm3(x))


# %% [output] Four heads: one distribution over 2,048 codes per codebook
class Heads(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.LayerNorm(c.hidden)
        self.heads = nn.ModuleList(nn.Linear(c.hidden, c.vocab, bias=False) for _ in range(c.codebooks))

    def forward(self, x):
        """(batch, steps, hidden) → logits (batch, codebooks, steps, vocab); the pad id is never predicted."""
        x = self.norm(x)
        return torch.stack([head(x) for head in self.heads], 1)


# %% [guidance] Classifier-free guidance: push the logits away from the no-prompt pass
def guide(cond, uncond, scale):
    """ℓ = ℓ(∅) + s · (ℓ(prompt) − ℓ(∅)); s = 1 is the prompted pass alone, s = 0 the unconditional one."""
    return uncond + scale * (cond - uncond)


# %% [sampling] Drawing a code from the top 250
def top_k(logits, k):
    """logits (vocab,) → the k most likely ids, most likely first, and their probabilities renormalised over the k."""
    order = torch.argsort(-logits.double(), stable=True)[:k]
    return order, torch.softmax(logits.double()[order], 0)


def draw(ids, probs, u):
    """The id whose slice of [0, 1), laid out most likely first, holds the uniform u."""
    edges = torch.cumsum(probs, 0)
    i = int(torch.searchsorted(edges, torch.tensor([u], dtype=edges.dtype), right=True)[0])
    return int(ids[min(i, len(ids) - 1)])


# %% [codec-decode] EnCodec: codes → the latent its decoder turns into 32 kHz audio
def codes_to_latent(codes, codebooks):
    """codes (k, frames), codebooks (4, 2048, 128) → (128, frames): the sum of each codebook's chosen vector. With
    fewer codebooks the sum stops early; EnCodec's convolutional decoder upsamples 8 · 5 · 4 · 4 = 640 times."""
    return sum(codebook[ids] for codebook, ids in zip(codebooks, codes)).T


# %% [model] The decoder: the codes so far and the prompt → the next step's logits
class MusicGenDecoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embedding = Embedding(c)
        self.text = TextConditioning(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.output = Heads(c)

    def forward(self, grid, t5_out, mask):
        """grid (batch, codebooks, steps) of the delayed codes, t5_out (batch, words, 768) → logits for the next
        step after every step: (batch, codebooks, steps, vocab)."""
        text = self.text(t5_out, mask)
        x = self.embedding(grid)
        for block in self.blocks:
            x = block(x, text, mask)
        return self.output(x)

    def guided(self, grid, t5_out, mask, scale):
        """One batch with the prompt and one with the null prompt, mixed: (codebooks, steps, vocab)."""
        both = self(grid.expand(2, -1, -1), torch.cat([t5_out, torch.zeros_like(t5_out)]),
                    torch.cat([mask, torch.zeros_like(mask)]))
        return guide(both[0], both[1], scale)
`;export{e as default};
