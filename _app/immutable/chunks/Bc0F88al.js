const e=`# T5-small, the model in the Encoder–decoder tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32128
    d_model: int = 512
    n_layers: int = 6  # in each stack
    n_heads: int = 8  # 64 dimensions per head
    d_mlp: int = 2048
    n_buckets: int = 32  # relative-position buckets
    max_distance: int = 128  # farther positions share the last bucket
    start_id: int = 0  # the decoder starts from the pad token
    eps: float = 1e-6


# %% [embedding] Embedding
class Embedding(nn.Module):
    """One token table, shared by the encoder, the decoder and the output projection. No position
    vector is added: positions enter as a learned bias inside self-attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids)  # (batch, seq, d_model)


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    """T5's LayerNorm: divide by the root mean square. No mean subtraction and no bias."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# %% [relative-bias] Relative position bias
def relative_bucket(distance, bidirectional, n_buckets, max_distance):
    """Bucket of key position − query position: one bucket per distance up to 7 (15 in the decoder),
    then log-spaced buckets up to max_distance."""
    bucket = torch.zeros_like(distance)
    if bidirectional:  # the encoder: half of the buckets are for keys after the query
        n_buckets //= 2
        bucket += (distance > 0).long() * n_buckets
        distance = distance.abs()
    else:  # the decoder only looks back
        distance = (-distance).clamp(min=0)
    exact = n_buckets // 2
    far = exact + (
        torch.log(distance.float() / exact) / math.log(max_distance / exact) * (n_buckets - exact)
    ).long()
    return bucket + torch.where(distance < exact, distance, far.clamp(max=n_buckets - 1))


class RelativeBias(nn.Module):
    """A learned scalar per bucket and head. Each stack computes it once, in its first block, and
    adds it to the attention scores of all its blocks."""

    def __init__(self, c: Config, bidirectional: bool):
        super().__init__()
        self.c, self.bidirectional = c, bidirectional
        self.embedding = nn.Embedding(c.n_buckets, c.n_heads)

    def forward(self, queries, keys):
        device = self.embedding.weight.device
        distance = torch.arange(keys, device=device) - torch.arange(queries, device=device)[:, None]
        bucket = relative_bucket(distance, self.bidirectional, self.c.n_buckets, self.c.max_distance)
        return self.embedding(bucket).permute(2, 0, 1)  # (heads, queries, keys)


# %% [attention] Multi-head self-attention
class Attention(nn.Module):
    """No biases in the projections and no 1/√d scaling of the scores: T5 folds that into the
    initialization of the query weights."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def heads(self, z):  # (batch, seq, d_model) → (batch, heads, seq, head_dim)
        return z.view(z.shape[0], z.shape[1], self.n_heads, -1).transpose(1, 2)

    def attend(self, x, memory, bias):
        q, k, v = self.heads(self.q(x)), self.heads(self.k(memory)), self.heads(self.v(memory))
        scores = q @ k.transpose(-2, -1) + bias  # (batch, heads, queries, keys)
        weights = scores.softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(x.shape))

    def forward(self, x, bias):  # bias: (heads, seq, seq); the decoder's has −inf above the diagonal
        return self.attend(x, x, bias)


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)  # wi
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)  # wo

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


# %% [encoder-block] Encoder block
class EncoderBlock(nn.Module):
    """Pre-norm: x + SelfAttention(RMSNorm(x)), then x + MLP(RMSNorm(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, bias):
        x = x + self.attention(self.norm1(x), bias)
        return x + self.mlp(self.norm2(x))


# %% [encoder] Encoder hidden state
class Encoder(nn.Module):
    """The encoder blocks share one bidirectional relative bias; a final RMSNorm gives the hidden
    state that every cross-attention reads."""

    def __init__(self, c: Config):
        super().__init__()
        self.relative_bias = RelativeBias(c, bidirectional=True)
        self.blocks = nn.ModuleList(EncoderBlock(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)

    def forward(self, x):  # x: embedded source tokens, (batch, source, d_model)
        bias = self.relative_bias(x.shape[1], x.shape[1])
        for block in self.blocks:
            x = block(x, bias)
        return self.norm(x)


# %% [cross-attention] Cross-attention
class CrossAttention(Attention):
    """Queries come from the decoder, keys and values from the encoder hidden state. No position bias
    and no mask: every target token sees every source token."""

    def forward(self, x, encoder_out):  # x: (batch, target, d), encoder_out: (batch, source, d)
        return self.attend(x, encoder_out, bias=0.0)


# %% [decoder-block] Decoder block
class DecoderBlock(nn.Module):
    """Masked self-attention, cross-attention, then the MLP, each pre-norm with a residual."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.cross_attention = CrossAttention(c)
        self.norm3 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, encoder_out, bias):
        x = x + self.attention(self.norm1(x), bias)
        x = x + self.cross_attention(self.norm2(x), encoder_out)
        return x + self.mlp(self.norm3(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm, then the shared token table as the projection. Because the table is shared, the
    input is first scaled by 1/√d_model."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding
        self.scale = c.d_model**-0.5

    def forward(self, x):
        return self.lm_head(self.norm(x) * self.scale)  # logits: (batch, seq, vocab)


def next_token_probabilities(logits, temperature=1.0, top_k=None, top_p=None):
    """The last position's logits, scaled by temperature and cut to top-k or top-p, as probabilities."""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        p = sorted_logits.softmax(dim=-1)
        drop = p.cumsum(dim=-1) - p >= top_p  # keep the smallest set whose mass reaches top_p
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits.softmax(dim=-1)


# %% [model] Full model
class T5(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.encoder = Encoder(c)
        self.decoder_bias = RelativeBias(c, bidirectional=False)
        self.decoder = nn.ModuleList(DecoderBlock(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, source_ids, target_ids):  # target_ids start with Config.start_id
        encoder_out = self.encoder(self.embedding(source_ids))
        x = self.embedding(target_ids)
        t = x.shape[1]
        future = torch.full((t, t), float("-inf"), device=x.device).triu(1)
        bias = self.decoder_bias(t, t) + future  # (heads, target, target)
        for block in self.decoder:
            x = block(x, encoder_out, bias)
        return self.output(x)


# source_ids = tokenizer("translate English to German: The house is wonderful.")
# probabilities = next_token_probabilities(T5()(source_ids, target_ids), temperature=0.8, top_k=5)
`;export{e as default};
