const e=`# wav2vec 2.0 base (Baevski et al. 2020) as plain PyTorch: facebook/wav2vec2-base-960h, the model the Audio
# Explainer's wav2vec 2.0 tab records, and the pretraining head of facebook/wav2vec2-base (same encoder shapes).
# Raw 16 kHz audio goes through seven 1-D convolutions to one 512-wide vector every 20 ms, a linear projection to
# 768, a convolutional position embedding and 12 post-norm transformer blocks. Fine-tuned with CTC, a linear head
# turns every frame into 32 log-probabilities (a blank, 26 letters, the apostrophe, the word boundary | and four
# unused specials); decoding collapses repeats and drops blanks. Pretrained, a product quantizer turns the
# convolutional features into targets and the transformer learns to pick the right one for masked frames.
# Each "# %% [id]" section is one step of the tab's diagram.
import math
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    sample_rate: int = 16_000
    conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple = (5, 2, 2, 2, 2, 2, 2)  # 5 × 2⁶ = 320 samples: one frame every 20 ms
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12  # 64 dimensions per head
    d_ff: int = 3072
    pos_kernel: int = 128  # the position convolution sees 128 frames, 2.56 s
    pos_groups: int = 16
    vocab_size: int = 32  # <pad> (the CTC blank) <s> </s> <unk> | E T A O N I H S R D L U M W C F G Y P B V K ' X J Q Z
    blank: int = 0
    word: int = 4  # "|", written as a space
    eps: float = 1e-5
    # Pretraining (facebook/wav2vec2-base)
    groups: int = 2  # product quantizer: one choice from each of 2 codebooks
    entries: int = 320  # 320 × 320 = 102,400 possible targets
    codevector_dim: int = 256  # 128 per group, concatenated
    proj_dim: int = 256  # both sides of the contrastive comparison
    temperature: float = 0.1  # κ in the paper
    diversity_weight: float = 0.1


# %% [feature-encoder] Convolutional feature encoder: 16 kHz samples → one vector every 20 ms
class FeatureEncoder(nn.Module):
    """Seven convolutions over time, each followed by GELU; the first is normalised per channel over the whole clip
    (GroupNorm with one group per channel). The strides multiply to 320 samples and the receptive field is 400
    samples, so frame t sees samples 320 t … 320 t + 399: 25 ms of audio every 20 ms."""

    def __init__(self, c: Config):
        super().__init__()
        dims = (1, *c.conv_dim)
        self.convs = nn.ModuleList(
            nn.Conv1d(dims[i], dims[i + 1], c.conv_kernel[i], stride=c.conv_stride[i], bias=False)
            for i in range(len(c.conv_dim))
        )
        self.norm = nn.GroupNorm(c.conv_dim[0], c.conv_dim[0])

    def forward(self, audio):  # audio: (batch, samples), zero mean and unit variance
        x = audio[:, None]
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.gelu(self.norm(x) if i == 0 else x)
        return x.transpose(1, 2)  # (batch, frames, 512)


def frames(samples, c: Config):
    """How many vectors the encoder writes: each convolution keeps ⌊(length − kernel) / stride⌋ + 1."""
    for kernel, stride in zip(c.conv_kernel, c.conv_stride):
        samples = (samples - kernel) // stride + 1
    return samples


def receptive_field(c: Config):
    """(samples one frame sees, samples between frames): 400 and 320 at 16 kHz."""
    size, hop = 1, 1
    for kernel, stride in zip(reversed(c.conv_kernel), reversed(c.conv_stride)):
        size = (size - 1) * stride + kernel
    for stride in c.conv_stride:
        hop *= stride
    return size, hop


# %% [feature-projection] LayerNorm, then 512 → 768
class FeatureProjection(nn.Module):
    """The normalised 512-wide features are also what the quantizer reads during pretraining."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.LayerNorm(c.conv_dim[-1], eps=c.eps)
        self.proj = nn.Linear(c.conv_dim[-1], c.d_model)

    def forward(self, features):
        normed = self.norm(features)
        return self.proj(normed), normed


# %% [position] Convolutional position embedding
class PositionConv(nn.Module):
    """No position table: a grouped convolution over 128 neighbouring frames (16 groups of 48 channels, weight
    normalised, padded to keep the length) adds where-am-I information, then GELU and a residual add and a
    LayerNorm. The weight is g · v / ‖v‖, with the norm taken per kernel tap."""

    def __init__(self, c: Config):
        super().__init__()
        self.pad = c.pos_kernel // 2
        self.groups = c.pos_groups
        self.weight_g = nn.Parameter(torch.ones(1, 1, c.pos_kernel))
        self.weight_v = nn.Parameter(torch.randn(c.d_model, c.d_model // c.pos_groups, c.pos_kernel))
        self.bias = nn.Parameter(torch.zeros(c.d_model))
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, x):  # (batch, frames, d_model)
        v = self.weight_v
        weight = self.weight_g * v / v.norm(dim=(0, 1), keepdim=True)
        y = F.conv1d(x.transpose(1, 2), weight, self.bias, padding=self.pad, groups=self.groups)
        y = F.gelu(y[..., : x.shape[1]])  # an even kernel writes one frame too many
        return self.norm(x + y.transpose(1, 2))


# %% [attention] Multi-head self-attention over every frame
class Attention(nn.Module):
    """No mask: every frame reads every other frame of the clip, before and after it."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, frames, d_model)
        b, t, d = x.shape
        q, k, v = (z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in (self.q(x), self.k(x), self.v(x)))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (batch, heads, t, t)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, d)), weights


# %% [block] Transformer block (post-norm)
class Block(nn.Module):
    """wav2vec 2.0 base normalises after each residual add: LN(x + Attention(x)), then LN(x + FFN(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.attention = Attention(c)
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.up = nn.Linear(c.d_model, c.d_ff)
        self.down = nn.Linear(c.d_ff, c.d_model)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, x):
        mixed, weights = self.attention(x)
        x = self.norm1(x + mixed)
        return self.norm2(x + self.down(F.gelu(self.up(x)))), weights


# %% [output] CTC head: 32 log-probabilities per frame
class CTCHead(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.proj = nn.Linear(c.d_model, c.vocab_size)

    def forward(self, x):  # (batch, frames, d_model) → (batch, frames, 32)
        return self.proj(x).log_softmax(dim=-1)


# %% [greedy] Greedy CTC decoding: best symbol per frame, collapse repeats, drop blanks
def greedy(log_probs, vocab, c: Config):
    """log_probs (frames, 32) → text. The best path keeps the likeliest symbol at every frame; a run of the same
    symbol is one letter, blanks separate letters (so a real double letter needs a blank between) and are dropped,
    and | becomes a space."""
    path = log_probs.argmax(dim=-1).tolist()
    collapsed = [s for i, s in enumerate(path) if i == 0 or s != path[i - 1]]
    letters = [s for s in collapsed if s != c.blank]
    return "".join(" " if s == c.word else vocab[s] for s in letters).strip()


# %% [beam] CTC prefix beam search
def logadd(*xs):
    top = max(xs)
    return top if top == -math.inf else top + math.log(sum(math.exp(x - top) for x in xs))


def beam_search(log_probs, c: Config, width=8, prune=-30.0):
    """Keeps the \`width\` likeliest prefixes (label sequences, already collapsed) frame by frame. Each prefix
    carries two log-probabilities, summed over every alignment that spells it: paths ending in a blank (p_b) and
    paths ending in its last letter (p_nb). Unlike the best path, a prefix's score adds up all its alignments (all
    of them while the beam keeps every prefix; pruning drops the ones that passed through a dropped prefix).
    Returns, after every frame, the beam as [(prefix ids, log p_b, log p_nb)], likeliest first."""
    beam = {(): (0.0, -math.inf)}
    history = []
    for row in log_probs.tolist():
        nxt = defaultdict(lambda: [-math.inf, -math.inf])
        for prefix, (pb, pnb) in beam.items():
            for s, q in enumerate(row):
                if q < prune:  # symbols this unlikely cannot change the beam
                    continue
                if s == c.blank:  # the prefix stays; now it ends in a blank
                    e = nxt[prefix]
                    e[0] = logadd(e[0], pb + q, pnb + q)
                    continue
                longer = prefix + (s,)
                if prefix and prefix[-1] == s:
                    e = nxt[longer]  # a repeated letter only counts after a blank
                    e[1] = logadd(e[1], pb + q)
                    e = nxt[prefix]  # otherwise it continues the same letter
                    e[1] = logadd(e[1], pnb + q)
                else:
                    e = nxt[longer]
                    e[1] = logadd(e[1], pb + q, pnb + q)
        alive = [item for item in nxt.items() if logadd(*item[1]) > -math.inf]  # e.g. "ll" in too few frames
        ranked = sorted(alive, key=lambda item: -logadd(*item[1]))[:width]
        beam = {prefix: tuple(scores) for prefix, scores in ranked}
        history.append([(list(prefix), pb, pnb) for prefix, (pb, pnb) in ranked])
    return history


def labelling_log_prob(log_probs, ids, c: Config):
    """log p(ids | audio), summed over every alignment exactly (the CTC forward algorithm)."""
    loss = F.ctc_loss(log_probs[:, None], torch.tensor([ids]), torch.tensor([log_probs.shape[0]]),
                      torch.tensor([len(ids)]), blank=c.blank, reduction="sum", zero_infinity=False)
    return -loss.item()


# %% [quantizer] Product quantizer: the pretraining targets
class Quantizer(nn.Module):
    """Pretraining only. Each normalised 512-wide feature picks one entry from each of two codebooks of 320 learned
    128-wide vectors (Gumbel softmax while training; the argmax here); the two are concatenated and projected to
    256. These targets come from the features before the transformer, so the model cannot copy them."""

    def __init__(self, c: Config):
        super().__init__()
        self.groups, self.entries = c.groups, c.entries
        self.logits = nn.Linear(c.conv_dim[-1], c.groups * c.entries)
        self.codebooks = nn.Parameter(torch.randn(c.groups, c.entries, c.codevector_dim // c.groups))
        self.proj = nn.Linear(c.codevector_dim, c.proj_dim)

    def forward(self, normed):  # (batch, frames, 512)
        b, t, _ = normed.shape
        logits = self.logits(normed).view(b, t, self.groups, self.entries)
        ids = logits.argmax(dim=-1)  # (batch, frames, groups)
        chosen = torch.stack([self.codebooks[g][ids[..., g]] for g in range(self.groups)], dim=2)
        return self.proj(chosen.reshape(b, t, -1)), ids


def perplexity(ids, mask, c: Config):
    """How evenly the masked frames use each codebook: exp(entropy) of each codebook's usage, summed over the
    two (at most 2 × 320). The diversity loss pushes it up."""
    total = 0.0
    for g in range(c.groups):
        counts = torch.bincount(ids[..., g][mask], minlength=c.entries).double()
        p = counts / counts.sum()
        total += torch.exp(-(p * torch.log(p + 1e-7)).sum())
    return total


# %% [contrastive] Contrastive task over masked frames
def contrastive(context, targets, mask, negatives, ids, c: Config):
    """context (frames, 256): the transformer's output for frames whose features were replaced by a learned mask
    vector, projected; targets (frames, 256): the quantized targets of every frame; negatives (frames, K): for each
    masked frame, K other masked frames of the clip whose targets serve as distractors. Logits are cosine
    similarities divided by κ, the true target first; the loss is cross-entropy towards it, summed over masked
    frames, plus the diversity term."""
    candidates = torch.cat([targets[:, None], targets[negatives]], dim=1)  # (frames, 1 + K, 256)
    logits = F.cosine_similarity(context[:, None], candidates, dim=-1) / c.temperature
    same = (targets[negatives] == targets[:, None]).all(-1)  # a distractor with the very same target is skipped
    logits[:, 1:] = logits[:, 1:].masked_fill(same, -math.inf)
    loss = F.cross_entropy(logits[mask], torch.zeros(int(mask.sum()), dtype=torch.long), reduction="sum")
    n = c.groups * c.entries
    diversity = (n - perplexity(ids, mask, c)) / n * mask.sum()
    return logits, loss, diversity, loss + c.diversity_weight * diversity


class Pretraining(nn.Module):
    """wav2vec2-base before fine-tuning: the same encoder, a learned vector that replaces masked frames' features
    before the position convolution, a projection of the transformer output to 256, and the quantizer."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.features = FeatureEncoder(c)
        self.projection = FeatureProjection(c)
        self.mask_embed = nn.Parameter(torch.zeros(c.d_model))
        self.position = PositionConv(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.project_context = nn.Linear(c.d_model, c.proj_dim)
        self.quantizer = Quantizer(c)

    def forward(self, audio, mask, negatives):  # audio (1, samples); mask (frames,) bool; negatives (frames, K)
        x, normed = self.projection(self.features(audio))
        x = torch.where(mask[None, :, None], self.mask_embed, x)  # hide the masked frames from the transformer
        x = self.position(x)
        for block in self.blocks:
            x, _ = block(x)
        targets, ids = self.quantizer(normed)  # from the unmasked features
        context = self.project_context(x)
        return contrastive(context[0], targets[0], mask, negatives, ids[0], self.c), ids[0]


# %% [model] wav2vec 2.0 for CTC
class Wav2Vec2ForCTC(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.features = FeatureEncoder(c)
        self.projection = FeatureProjection(c)
        self.position = PositionConv(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.head = CTCHead(c)

    def forward(self, audio):  # audio: (batch, samples) at 16 kHz, normalised to zero mean and unit variance
        x, _ = self.projection(self.features(audio))
        x = self.position(x)
        for block in self.blocks:
            x, _ = block(x)
        return self.head(x)  # (batch, frames, 32): log-probabilities for CTC
`;export{e as default};
