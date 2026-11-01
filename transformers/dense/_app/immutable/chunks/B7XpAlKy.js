const e=`# A plain recurrent network (Elman 1990), character level, trained locally on Tiny Shakespeare with the LSTM tab's
# sizes and settings. The network reads one character at a time and keeps one vector, its hidden state h. Each step
# squashes the new character and the previous state through one tanh: h_t = tanh(W_x x_t + W_h h_{t−1} + b). Nothing
# else carries information forward, so everything it remembers has to survive that matrix and that tanh at every
# step, and so does every gradient on its way back: it shrinks (or grows) by the same kind of factor at each step.
# Each "# %% [id]" section is one part of the RNN tab. Weights are laid out as in torch.nn.RNN.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65  # the distinct characters of Tiny Shakespeare
    embed_dim: int = 64
    hidden: int = 256  # units in h, as many as the LSTM tab's h and c
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


# %% [step] One step: h_t = tanh(W_x x_t + W_h h_{t−1} + b)
def rnn_step(x, h_prev, W_ih, W_hh, b_ih, b_hh):
    """x: (embed_dim,), h_prev: (hidden,); W_ih: (hidden, embed_dim) reads the character, W_hh: (hidden, hidden) the
    previous state. nn.RNN keeps two biases; they add up to the one b of the equation."""
    from_x = W_ih @ x + b_ih
    from_h = W_hh @ h_prev + b_hh
    return torch.tanh(from_x + from_h), from_x, from_h  # h in (−1, 1): the whole memory, rewritten every step


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


# %% [bptt] Backpropagation through time
@torch.no_grad()
def hidden_grads(h, p_last, target, W_hh, W_out):
    """The gradient of the last character's loss, L = −log p_T(target), with respect to every earlier hidden state.
    h: (T, hidden), the states the forward pass produced; p_last: the last step's next-character distribution.
    Going back one step multiplies by the step's Jacobian ∂h_k/∂h_{k−1} = diag(1 − h_k²) W_hh, transposed."""
    grad = W_out.T @ (p_last - F.one_hot(target, p_last.shape[-1]).to(p_last))  # ∂L/∂h_T
    grads = [grad]
    for k in range(len(h) - 1, 0, -1):
        grad = W_hh.T @ ((1 - h[k] ** 2) * grad)  # ∂L/∂h_{k−1}
        grads.append(grad)
    return torch.stack(grads[::-1])  # (T, hidden): row k is ∂L/∂h_k


def step_jacobian_norm(h_k, W_hh):
    """‖∂h_k/∂h_{k−1}‖₂, the most one step can stretch a gradient. Below 1 at every step, gradients vanish
    geometrically; tanh' = 1 − h² is near 0 for saturated units, which shrinks it further."""
    return torch.linalg.matrix_norm((1 - h_k[:, None] ** 2) * W_hh, ord=2)


# %% [model] The network and one training step
class CharRNN(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.embed_dim)
        self.rnn = nn.RNN(c.embed_dim, c.hidden, nonlinearity="tanh", batch_first=True)  # rnn_step, every step
        self.out = nn.Linear(c.hidden, c.vocab_size)

    def forward(self, ids, state=None):
        """ids: (batch, T) → logits (batch, T, vocab) and the final h to carry into the next chunk."""
        h, state = self.rnn(self.embed(ids), state)
        return self.out(h), state


def unrolled(model, ids):
    """The same forward pass written out with rnn_step, one character at a time, keeping both halves of each sum."""
    W_ih, W_hh = model.rnn.weight_ih_l0, model.rnn.weight_hh_l0
    b_ih, b_hh = model.rnn.bias_ih_l0, model.rnn.bias_hh_l0
    h = torch.zeros(model.rnn.hidden_size)
    trace = []
    for x in embed(model.embed.weight, ids):
        h, from_x, from_h = rnn_step(x, h, W_ih, W_hh, b_ih, b_hh)
        trace.append({"x": x, "fromX": from_x, "fromH": from_h, "h": h,
                      "p": next_char_probs(h, model.out.weight, model.out.bias)})
    return trace


def train_step(model, optimizer, batch, c: Config):
    """batch: (B, T + 1) character ids. Predict every next character; cross-entropy; clip; Adam (the LSTM tab's)."""
    inputs, targets = batch[:, :-1], batch[:, 1:]
    logits, _ = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
    optimizer.step()
    return loss.detach(), norm
`;export{e as default};
