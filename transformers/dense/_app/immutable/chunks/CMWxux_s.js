const e=`# Supervised fine-tuning (SFT), SmolLM2-135M → SmolLM2-135M-Instruct: the Post-training Explainer's SFT tab as plain
# PyTorch. A conversation is wrapped in the chat template, every token except the assistant's is masked out of the
# loss, the rest is scored with next-token cross-entropy, and one AdamW step moves every weight.
# \`model\` is any causal LM that maps ids (batch, seq) to logits (batch, seq, vocab), here SmolLM2 (a Llama decoder).
from bisect import bisect_right
from dataclasses import dataclass
from itertools import accumulate

import torch
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 49152
    ignore_index: int = -100  # the label of a token the loss skips
    default_system: str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    lr: float = 5e-5  # the tab's run; SmolLM2's recipe used 1e-3 with cosine decay over 2 epochs
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


# %% [chat-template] Chat template
def turns(messages, c: Config):
    """SmolLM2's template: the default system turn when there is none, then
    <|im_start|>role↵ content <|im_end|>↵ for every message, as (role, part, text) pieces."""
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": c.default_system}] + messages
    for m in messages:
        yield m["role"], "header", f"<|im_start|>{m['role']}\\n"
        yield m["role"], "content", m["content"]
        yield m["role"], "end", "<|im_end|>"
        yield m["role"], "newline", "\\n"


def encode(messages, tokenizer, c: Config):
    """Token ids of the whole conversation, and for each token the (role, part) where it starts. The text is
    tokenized in one go: a newline and the spaces after it can merge into one token across a piece boundary."""
    pieces = list(turns(messages, c))
    text = "".join(piece for *_, piece in pieces)
    ends = list(accumulate(len(piece) for *_, piece in pieces))  # character offset where each piece ends
    out = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    where = [pieces[bisect_right(ends, start)][:2] for start, _ in out["offset_mapping"]]
    return torch.tensor(out["input_ids"]), where


# %% [loss-mask] Loss mask
def labels_for(ids, where, c: Config):
    """Keep the assistant's content and its closing <|im_end|> (so the model learns to stop); every other
    token is still read as context but gets the label −100, which the loss skips."""
    scored = torch.tensor([role == "assistant" and part in ("content", "end") for role, part in where])
    return torch.where(scored, ids, torch.full_like(ids, c.ignore_index))


# %% [loss] Next-token cross-entropy
def per_token_loss(logits, labels, c: Config):
    """Position j − 1 predicts token j: shift by one, then −log p(token) wherever the label is kept."""
    logits, targets = logits[..., :-1, :], labels[..., 1:]  # (batch, seq − 1, vocab), (batch, seq − 1)
    keep = targets != c.ignore_index
    log_p = F.log_softmax(logits.float(), dim=-1)
    nll = -log_p.gather(-1, targets.clamp_min(0)[..., None])[..., 0]
    return torch.where(keep, nll, torch.zeros_like(nll)), keep


def sft_loss(logits, labels, c: Config):
    """The mean over scored tokens: L = Σ m_j · (−log p(x_j | x_<j)) / Σ m_j."""
    nll, keep = per_token_loss(logits, labels, c)
    return nll.sum() / keep.sum()


# %% [sft-step] One SFT step (AdamW)
class AdamW:
    """m ← β₁m + (1 − β₁)g, v ← β₂v + (1 − β₂)g², w ← w − lr·(m̂ / (√v̂ + ε) + λw), with m̂, v̂ bias-corrected."""

    def __init__(self, params, c: Config):
        self.params, self.c, self.t = list(params), c, 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        (b1, b2), lr = self.c.betas, self.c.lr
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            p.mul_(1 - lr * self.c.weight_decay)  # decoupled weight decay (0 in this run)
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            m_hat, v_hat = m / (1 - b1**self.t), v / (1 - b2**self.t)
            p.sub_(lr * m_hat / (v_hat.sqrt() + self.c.eps))


def sft_step(model, optimizer, messages, tokenizer, c: Config):
    """Encode, mask, score, backpropagate, update every weight; returns the loss before the update."""
    ids, where = encode(messages, tokenizer, c)
    labels = labels_for(ids, where, c)
    loss = sft_loss(model(ids[None]), labels[None], c)
    for p in optimizer.params:
        p.grad = None
    loss.backward()
    optimizer.step()
    return loss.detach()
`;export{e as default};
