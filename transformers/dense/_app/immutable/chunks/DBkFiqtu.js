const e=`# A character-level LSTM (Hochreiter & Schmidhuber 1997; forget gate: Gers et al. 2000), trained locally on Tiny
# Shakespeare. The network reads one character at a time and keeps two vectors: a hidden state h, which it shows to
# the output layer, and a cell state c, its memory. At every step four gates computed from the character and the
# previous h decide how much of the memory to keep (forget gate f), what to write (input gate i times candidate g)
# and how much of it to show (output gate o). The cell is updated by multiplying and adding, not by a squashing
# matrix product, so information and gradients can travel across many steps.
# Each "# %% [id]" section is one part of the LSTM tab. Weights are laid out as in torch.nn.LSTM (gates i, f, g, o).
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65  # the distinct characters of Tiny Shakespeare
    embed_dim: int = 64
    hidden: int = 256  # units in h and in c
    seq_len: int = 128  # characters per training sequence (truncated backpropagation through time)
    batch: int = 64
    steps: int = 4000
    lr: float = 2e-3
    clip: float = 1.0  # gradient norm clipping, the usual guard against exploding gradients in RNNs
    seed: int = 0


# %% [embedding] Character → vector
def embed(table, ids):
    """table: (vocab, embed_dim), a learned row per character; ids: (T,) → x: (T, embed_dim)."""
    return table[ids]


# %% [gates] The four gates from x_t and h_{t−1}
def gates(x, h, W_ih, W_hh, b_ih, b_hh):
    """x: (embed_dim,), h: (hidden,); W_ih: (4·hidden, embed_dim), W_hh: (4·hidden, hidden).
    One matrix product gives all four pre-activations, stacked in nn.LSTM's order i, f, g, o."""
    pre = W_ih @ x + b_ih + W_hh @ h + b_hh
    i, f, g, o = pre.chunk(4)
    return torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
    # i, f, o in (0, 1) are gates; g in (−1, 1) is the candidate the input gate lets in


# %% [cell] The memory cell and the hidden state
def cell(i, f, g, o, c_prev):
    """c_t = f ⊙ c_{t−1} + i ⊙ g: keep a fraction f of each memory unit and add the gated candidate.
    h_t = o ⊙ tanh(c_t): what the rest of the network sees of the memory."""
    c = f * c_prev + i * g
    h = o * torch.tanh(c)
    return c, h


def lstm_step(x, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh):
    i, f, g, o = gates(x, h_prev, W_ih, W_hh, b_ih, b_hh)
    c, h = cell(i, f, g, o, c_prev)
    return (i, f, g, o), c, h


# %% [output] Next-character distribution and sampling
def next_char_probs(h, W_out, b_out, temperature=1.0):
    """A linear layer over the 65 characters, then a softmax. Temperature < 1 sharpens the distribution."""
    return F.softmax((W_out @ h + b_out) / temperature, dim=-1)


def sample(model, start_ids, length, temperature, generator):
    """Feed the start text, then repeatedly draw a character from the predicted distribution and feed it back."""
    with torch.no_grad():
        logits, state = model(start_ids[None])
        out = []
        for _ in range(length):
            p = F.softmax(logits[0, -1] / temperature, dim=-1)
            nxt = torch.multinomial(p, 1, generator=generator)
            out.append(int(nxt))
            logits, state = model(nxt[None], state)
    return out


# %% [model] The network and one training step
class CharLSTM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.embed_dim)
        self.lstm = nn.LSTM(c.embed_dim, c.hidden, batch_first=True)  # the gates and cell above, for every step
        self.out = nn.Linear(c.hidden, c.vocab_size)

    def forward(self, ids, state=None):
        """ids: (batch, T) → logits (batch, T, vocab) and the final (h, c) to carry into the next chunk."""
        h, state = self.lstm(self.embed(ids), state)
        return self.out(h), state


def unrolled(model, ids):
    """The same forward pass written out with lstm_step, one character at a time, keeping every gate."""
    W_ih, W_hh = model.lstm.weight_ih_l0, model.lstm.weight_hh_l0
    b_ih, b_hh = model.lstm.bias_ih_l0, model.lstm.bias_hh_l0
    h = c = torch.zeros(model.lstm.hidden_size)
    trace = []
    for x in embed(model.embed.weight, ids):
        (i, f, g, o), c, h = lstm_step(x, h, c, W_ih, W_hh, b_ih, b_hh)
        trace.append({"i": i, "f": f, "g": g, "o": o, "c": c, "h": h,
                      "p": next_char_probs(h, model.out.weight, model.out.bias)})
    return trace


def train_step(model, optimizer, batch, c: Config):
    """batch: (B, T + 1) character ids. Predict every next character; cross-entropy; clip; Adam."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits, _ = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
    optimizer.step()
    return loss.detach(), norm
`;export{e as default};
