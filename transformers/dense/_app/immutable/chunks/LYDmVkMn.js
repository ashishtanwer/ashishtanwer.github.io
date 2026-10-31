const e=`# The logit lens (nostalgebraist 2020) on GPT-2 small, the logit-lens tab, as plain PyTorch.
# GPT-2 adds every block's output to one residual stream and reads its prediction off the end of it: final
# LayerNorm, then the unembedding (the token embedding matrix, transposed). The logit lens applies that same readout
# to the stream after every block, as if the model stopped there, and asks how close each early reading is to the
# answer the full model gives. Weights: openai-community/gpt2 (GPT2LMHeadModel); each "# %% [id]" section is one card.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    layers: int = 12
    heads: int = 12
    d_model: int = 768
    d_head: int = 64
    vocab: int = 50257
    ln_eps: float = 1e-5
    top_k: int = 5


# %% [residual] The residual stream after every block
def residual_stream(model, ids, c: Config = Config()):
    """ids: (T,) → (layers + 1, T, d_model): the embeddings, then the stream after blocks 1 … 12.
    Each block adds attention and then an MLP to the stream; nothing else ever writes to it."""
    tr = model.transformer
    T = ids.shape[0]
    h = tr.wte.weight[ids] + tr.wpe.weight[:T]  # token + position embedding
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    stream = [h]
    for block in tr.h:
        x = F.layer_norm(h, (c.d_model,), block.ln_1.weight, block.ln_1.bias, c.ln_eps)
        q, k, v = (x @ block.attn.c_attn.weight + block.attn.c_attn.bias).split(c.d_model, dim=-1)
        q, k, v = (t.view(T, c.heads, c.d_head).transpose(0, 1) for t in (q, k, v))  # (heads, T, d_head)
        scores = (q @ k.transpose(-1, -2) / c.d_head**0.5).masked_fill(~causal, float("-inf"))
        z = (scores.softmax(-1) @ v).transpose(0, 1).reshape(T, c.d_model)  # heads side by side
        h = h + z @ block.attn.c_proj.weight + block.attn.c_proj.bias
        x = F.layer_norm(h, (c.d_model,), block.ln_2.weight, block.ln_2.bias, c.ln_eps)
        hidden = F.gelu(x @ block.mlp.c_fc.weight + block.mlp.c_fc.bias, approximate="tanh")
        h = h + hidden @ block.mlp.c_proj.weight + block.mlp.c_proj.bias
        stream.append(h)
    return torch.stack(stream)


# %% [lens] Reading the stream through the unembedding
def logit_lens(model, h, c: Config = Config()):
    """h: (..., d_model), the stream at any point → (..., vocab) logits, exactly as the model reads its last point:
    the final LayerNorm (centre, divide by the standard deviation, scale by γ, shift by β) and the tied unembedding."""
    ln = model.transformer.ln_f
    mu = h.mean(-1, keepdim=True)
    sigma = (h.var(-1, unbiased=False, keepdim=True) + c.ln_eps).sqrt()  # one scale per position
    normed = (h - mu) / sigma * ln.weight + ln.bias
    return normed @ model.transformer.wte.weight.T  # W_U = W_Eᵀ


# %% [readout] Probability, rank and KL of every reading
def readout(lens_logits, final_logits, target, c: Config = Config()):
    """lens_logits, final_logits: (..., vocab); target: (...) the correct next token. In float64:
    - the lens's top-k tokens and their probabilities, and its entropy;
    - the correct token's probability and rank (1 = the lens's favourite);
    - KL(final ‖ lens): how many nats the reading is from the model's own output distribution."""
    logp = lens_logits.double().log_softmax(-1)
    final = final_logits.double().log_softmax(-1)
    top_p, top_ids = logp.exp().topk(c.top_k, dim=-1)
    target_logp = logp.gather(-1, target[..., None])[..., 0]
    return {
        "top_ids": top_ids,
        "top_p": top_p,
        "target_p": target_logp.exp(),
        "target_rank": (logp > target_logp[..., None]).sum(-1) + 1,
        "entropy": -(logp.exp() * logp).sum(-1),
        "kl": (final.exp() * (final - logp)).sum(-1),
    }


# %% [settle] Where the prediction settles
def settle_rows(top_ids):
    """top_ids: (layers + 1, T) the lens's favourite token per row and position → (T,) the first row from which the
    favourite is already the model's final answer and stays so up to the output."""
    agree = top_ids == top_ids[-1]
    rows = []
    for t in range(top_ids.shape[1]):
        r = top_ids.shape[0] - 1
        while r > 0 and agree[r - 1, t]:
            r -= 1
        rows.append(r)
    return torch.tensor(rows)


# %% [model] The whole lens
@torch.no_grad()
def lens_table(model, ids, c: Config = Config()):
    """Every position of a prompt but the last predicts the next token of the text: read all 13 points there."""
    stream = residual_stream(model, ids, c)[:, :-1]  # (13, T - 1, d)
    logits = logit_lens(model, stream, c)
    table = readout(logits, logits[-1].expand_as(logits), ids[1:].expand(stream.shape[0], -1), c)
    table["settle"] = settle_rows(table["top_ids"][..., 0])
    return table


# model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
# ids = torch.tensor(tokenizer("The Eiffel Tower is located in the city of Paris").input_ids)
# table = lens_table(model, ids)  # table["target_rank"][:, -1]: rank of " Paris" after each block
`;export{e as default};
