const e=`# An encoder–decoder with additive attention (Bahdanau, Cho & Bengio 2014, "Neural Machine Translation by Jointly
# Learning to Align and Translate"), trained locally on English → French sentence pairs from Tatoeba.
# A bidirectional GRU reads the source sentence and leaves one annotation per word (forward and backward states side
# by side). The decoder writes the translation one word at a time; before each word it scores every annotation
# against its own previous state, softmaxes the scores into weights, and reads the weighted average (the context).
# The weights over all output steps form the alignment matrix. Transformers kept this read-by-weights step and
# replaced the recurrence and the additive score with scaled dot products.
# Each "# %% [id]" section is one part of the tab. GRU weights are laid out as in torch.nn.GRU (gates r, z, n).
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    src_vocab: int = 4303  # English words seen at least twice in training, plus <pad> <bos> <eos> <unk>
    tgt_vocab: int = 5700  # French words, likewise
    embed_dim: int = 128
    enc_hidden: int = 128  # per direction; an annotation is 2 × 128
    dec_hidden: int = 256
    attn_dim: int = 128  # width of the additive score's hidden layer
    max_len: int = 16  # longest translation greedy decoding writes
    pad: int = 0
    bos: int = 1
    eos: int = 2
    unk: int = 3


# %% [encoder] Encoder: a bidirectional GRU leaves one annotation per source word
def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    """One GRU step (torch.nn.GRU's equations). r: how much of h the candidate may read; z: how much of h to keep."""
    gi, gh = W_ih @ x + b_ih, W_hh @ h + b_hh
    r = torch.sigmoid(gi.chunk(3)[0] + gh.chunk(3)[0])
    z = torch.sigmoid(gi.chunk(3)[1] + gh.chunk(3)[1])
    n = torch.tanh(gi.chunk(3)[2] + r * gh.chunk(3)[2])
    return (1 - z) * n + z * h


def encode(x, fwd, bwd):
    """x: (S, embed_dim) source word vectors; fwd, bwd: (W_ih, W_hh, b_ih, b_hh) of each direction.
    The forward GRU reads left to right, the backward GRU right to left; annotation j is both states at word j."""
    S, H = x.shape[0], fwd[1].shape[1]
    h_f, h_b = torch.zeros(H), torch.zeros(H)
    forward, backward = [], [None] * S
    for j in range(S):
        h_f = gru_cell(x[j], h_f, *fwd)
        forward.append(h_f)
    for j in reversed(range(S)):
        h_b = gru_cell(x[j], h_b, *bwd)
        backward[j] = h_b
    return torch.cat([torch.stack(forward), torch.stack(backward)], dim=-1)  # (S, 2H)


# %% [attention] Additive attention: score, weights, context
def additive_scores(s_prev, keys, W_a, b_a, v_a):
    """e_j = v_a · tanh(W_a s_{t−1} + b_a + U_a h_j). keys = U_a h_j is the same for every output step, so it is
    computed once per sentence; W_a s_{t−1} + b_a is the decoder's "query". A one-hidden-layer network scores each
    pair."""
    return torch.tanh(W_a @ s_prev + b_a + keys) @ v_a  # (S,)


def attend(s_prev, annotations, keys, W_a, b_a, v_a):
    """Weights α = softmax(e) over the source words, and the context c_t = Σ_j α_j h_j."""
    alpha = F.softmax(additive_scores(s_prev, keys, W_a, b_a, v_a), dim=-1)
    return alpha, alpha @ annotations  # (S,), (2H,)


# %% [decoder] One decoder step: attend, update the state, predict the next word
def decoder_step(y_prev, s_prev, annotations, keys, p):
    """y_prev: the previous output word's vector; s_prev: the decoder state. p holds the decoder's weights.
    The context is computed from s_{t−1}, then the GRU reads [y_{t−1}; c_t], then a linear layer over
    [s_t; c_t; y_{t−1}] gives the next word's logits."""
    alpha, context = attend(s_prev, annotations, keys, p["W_a"], p["b_a"], p["v_a"])
    s = gru_cell(torch.cat([y_prev, context]), s_prev, p["W_ih"], p["W_hh"], p["b_ih"], p["b_hh"])
    logits = p["W_out"] @ torch.cat([s, context, y_prev]) + p["b_out"]
    return alpha, context, s, logits


# %% [translate] Greedy translation
def translate(model, src_ids, c: Config):
    """Encode once, start from s_0 = tanh(W_s ←h_1) and <bos>, and keep the most likely word until <eos>."""
    p = model.decoder_params()
    with torch.no_grad():
        annotations = model.encode(src_ids[None])[0]
        keys = annotations @ p["U_a"].T
        s = torch.tanh(model.init_state(annotations[0, c.enc_hidden:]))  # s_0 from ←h_1
        y, out, steps = c.bos, [], []
        for _ in range(c.max_len):
            alpha, context, s_new, logits = decoder_step(model.tgt_embed.weight[y], s, annotations, keys, p)
            steps.append({"s_prev": s, "alpha": alpha, "context": context, "logits": logits})
            s, y = s_new, int(logits.argmax())
            if y == c.eos:
                break
            out.append(y)
    return out, steps


# %% [model] The network and one training step
class Seq2SeqAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.src_embed = nn.Embedding(c.src_vocab, c.embed_dim, padding_idx=c.pad)
        self.tgt_embed = nn.Embedding(c.tgt_vocab, c.embed_dim, padding_idx=c.pad)
        self.encoder = nn.GRU(c.embed_dim, c.enc_hidden, batch_first=True, bidirectional=True)
        self.init_state = nn.Linear(c.enc_hidden, c.dec_hidden)  # W_s on the backward state of the first word
        self.U_a = nn.Linear(2 * c.enc_hidden, c.attn_dim, bias=False)
        self.W_a = nn.Linear(c.dec_hidden, c.attn_dim)  # its bias is the score's hidden-layer bias
        self.v_a = nn.Linear(c.attn_dim, 1, bias=False)
        self.cell = nn.GRUCell(c.embed_dim + 2 * c.enc_hidden, c.dec_hidden)
        self.out = nn.Linear(c.dec_hidden + 2 * c.enc_hidden + c.embed_dim, c.tgt_vocab)
        self.c = c

    def encode(self, src):
        """src: (B, S) → annotations (B, S, 2H). Padding sits at the end of each row; packing makes each direction
        read only the real words (the backward GRU starts at the last word), and attention masks the padding."""
        lengths = (src != self.c.pad).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.src_embed(src), lengths, batch_first=True,
                                                   enforce_sorted=False)
        annotations, _ = nn.utils.rnn.pad_packed_sequence(self.encoder(packed)[0], batch_first=True,
                                                          total_length=src.shape[1])
        return annotations

    def forward(self, src, tgt_in):
        """Teacher forcing: the decoder reads the reference translation shifted right. Returns (B, T, vocab) logits
        and the (B, T, S) attention weights."""
        mask = src != self.c.pad
        annotations = self.encode(src)
        first_backward = annotations[:, 0, self.c.enc_hidden:]  # ←h_1, the backward GRU's last state
        s = torch.tanh(self.init_state(first_backward))
        keys = self.U_a(annotations)  # (B, S, A)
        logits, weights = [], []
        for y in self.tgt_embed(tgt_in).unbind(1):
            e = self.v_a(torch.tanh(self.W_a(s)[:, None] + keys)).squeeze(-1)
            alpha = F.softmax(e.masked_fill(~mask, -torch.inf), dim=-1)
            context = (alpha[:, :, None] * annotations).sum(1)
            s = self.cell(torch.cat([y, context], -1), s)
            logits.append(self.out(torch.cat([s, context, y], -1)))
            weights.append(alpha)
        return torch.stack(logits, 1), torch.stack(weights, 1)

    def decoder_params(self):
        return {"W_a": self.W_a.weight, "b_a": self.W_a.bias, "U_a": self.U_a.weight, "v_a": self.v_a.weight[0],
                "W_ih": self.cell.weight_ih, "W_hh": self.cell.weight_hh, "b_ih": self.cell.bias_ih,
                "b_hh": self.cell.bias_hh, "W_out": self.out.weight, "b_out": self.out.bias}


def train_step(model, optimizer, src, tgt, c: Config, clip=1.0):
    """src: (B, S); tgt: (B, T + 1) = <bos> … <eos>, padded. Cross-entropy on every real target word."""
    logits, _ = model(src, tgt[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1), ignore_index=c.pad)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss.detach()
`;export{e as default};
