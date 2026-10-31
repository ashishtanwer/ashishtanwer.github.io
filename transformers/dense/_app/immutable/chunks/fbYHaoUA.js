const e=`# Direct logit attribution on GPT-2 small, the attribution tab, as plain PyTorch.
# GPT-2's last-position stream is a plain sum: token embedding + position embedding + the output of every attention
# head + every attention layer's output bias + every MLP's output. Its next-token logits are W_E · LayerNorm_f(stream).
# LayerNorm is not linear, but with its scale σ frozen at its value on this prompt it is affine, so each term's share
# of a logit can be read off on its own, and the shares add up to the logit exactly. The tab reads the shares of
# logit(indirect object) − logit(subject) on indirect-object prompts ("When Mary and John went to the store, John gave
# a drink to" → " Mary" vs " John"). Weights: openai-community/gpt2 (GPT2LMHeadModel).
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    layers: int = 12
    heads: int = 12
    d_model: int = 768
    d_head: int = 64
    ln_eps: float = 1e-5


# %% [cache] One forward, keeping every term
@torch.no_grad()
def run_with_cache(model, ids, c: Config = Config()):
    """ids: (T,) → the logits and, per layer, each head's output z (heads side by side, before W_O), the attention
    pattern, the attention and MLP outputs, and the stream entering the final LayerNorm."""
    tr = model.transformer
    T = ids.shape[0]
    h = tr.wte.weight[ids] + tr.wpe.weight[:T]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    cache = {"z": [], "pattern": [], "attn": [], "mlp": []}
    for block in tr.h:
        x = F.layer_norm(h, (c.d_model,), block.ln_1.weight, block.ln_1.bias, c.ln_eps)
        q, k, v = (x @ block.attn.c_attn.weight + block.attn.c_attn.bias).split(c.d_model, dim=-1)
        q, k, v = (t.view(T, c.heads, c.d_head).transpose(0, 1) for t in (q, k, v))
        pattern = (q @ k.transpose(-1, -2) / c.d_head**0.5).masked_fill(~causal, float("-inf")).softmax(-1)
        z = (pattern @ v).transpose(0, 1).reshape(T, c.d_model)
        attn = z @ block.attn.c_proj.weight + block.attn.c_proj.bias
        h = h + attn
        x = F.layer_norm(h, (c.d_model,), block.ln_2.weight, block.ln_2.bias, c.ln_eps)
        mlp = F.gelu(x @ block.mlp.c_fc.weight + block.mlp.c_fc.bias, approximate="tanh") @ block.mlp.c_proj.weight
        mlp = mlp + block.mlp.c_proj.bias
        h = h + mlp
        for name, value in (("z", z), ("pattern", pattern), ("attn", attn), ("mlp", mlp)):
            cache[name].append(value)
    cache = {name: torch.stack(values) for name, values in cache.items()}
    cache["final"] = h
    ln = tr.ln_f
    cache["logits"] = F.layer_norm(h, (c.d_model,), ln.weight, ln.bias, c.ln_eps) @ tr.wte.weight.T
    return cache


# %% [heads] Splitting attention into heads
def head_outputs(z, block, c: Config = Config()):
    """z: (d_model,) one position's head outputs side by side → (heads, d_model): what each head writes to the stream.
    W_O is one (768, 768) matrix; head h owns its rows h·64 … h·64 + 63. The output bias belongs to no head."""
    W_O = block.attn.c_proj.weight.view(c.heads, c.d_head, c.d_model)
    return torch.einsum("hd,hdm->hm", z.view(c.heads, c.d_head), W_O)


# %% [frozen-norm] The final LayerNorm with its scale frozen
def frozen_scale(final, c: Config = Config()):
    """σ = √(var(h) + ε) of the actual stream entering the final LayerNorm, at this position."""
    return (final.var(-1, unbiased=False) + c.ln_eps).sqrt()


def share(term, direction, gamma, sigma):
    """A term's share of a logit (or of a difference of logits, along W_E[a] − W_E[b]): LayerNorm_f minus its β is
    linear once σ is fixed: centre the term, scale by γ / σ, read along the token's unembedding row."""
    return ((term - term.mean(-1, keepdim=True)) * gamma / sigma) @ direction


# %% [attribution] Every term's share of the logit difference
@torch.no_grad()
def attribute(model, cache, ids, io, s):
    """At the last position, the share of logit(io) − logit(s) of every term of the stream."""
    tr = model.transformer
    t = ids.shape[0] - 1
    direction = tr.wte.weight[io] - tr.wte.weight[s]
    gamma, sigma = tr.ln_f.weight, frozen_scale(cache["final"][t])
    return {
        "embedToken": share(tr.wte.weight[ids[t]], direction, gamma, sigma),
        "embedPos": share(tr.wpe.weight[t], direction, gamma, sigma),
        "heads": torch.stack([share(head_outputs(cache["z"][layer, t], block), direction, gamma, sigma)
                              for layer, block in enumerate(tr.h)]),  # (layers, heads)
        "attnBias": torch.stack([share(block.attn.c_proj.bias, direction, gamma, sigma) for block in tr.h]),
        "mlp": share(cache["mlp"][:, t], direction, gamma, sigma),  # (layers,)
        "lnBias": tr.ln_f.bias @ direction,  # β · (W_E[io] − W_E[s]): part of every logit difference
    }


# %% [model] Adding the shares up
@torch.no_grad()
def logit_difference(model, ids, io, s):
    """The shares and their total, which is the model's own logit difference at the last position."""
    cache = run_with_cache(model, ids)
    shares = attribute(model, cache, ids, io, s)
    total = sum(value.double().sum() for value in shares.values())
    actual = cache["logits"][-1, io] - cache["logits"][-1, s]
    return shares, total, actual


# model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
# ids = torch.tensor(tokenizer("When Mary and John went to the store, John gave a drink to").input_ids)
# shares, total, actual = logit_difference(model, ids, io=tokenizer.encode(" Mary")[0], s=tokenizer.encode(" John")[0])
`;export{e as default};
