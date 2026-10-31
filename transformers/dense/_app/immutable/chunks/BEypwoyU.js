const e=`# Activation patching on GPT-2 small, the patching tab, as plain PyTorch.
# Run two prompts that differ in one word: a clean one ("When Mary and John went to the store, John gave a drink to",
# answer " Mary") and a corrupted one ("…, Mary gave a drink to", where the answer flips to " John"). Then run the
# corrupted prompt again with one activation taken from the clean run. If that one activation brings the clean answer
# back, it carries the information the corruption destroyed. The metric is logit(" Mary") − logit(" John") at the last
# position; the fraction recovered is (patched − corrupted) / (clean − corrupted). Weights: openai-community/gpt2.
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


# %% [forward] GPT-2 with patch points
@torch.no_grad()
def run(model, ids, patch=None, c: Config = Config()):
    """ids: (batch, T). patch maps (kind, layer) → fn(value) → value, applied where the value is made:
    "resid" at points 0 … 12 (the embeddings, then the stream after each block), "z" (each head's output, heads side
    by side), "attn" and "mlp" (each layer's outputs). Returns the last position's logits and the clean values."""
    patch = patch or {}
    tr = model.transformer
    B, T = ids.shape
    keep = {}

    def point(kind, layer, value):
        if (kind, layer) in patch:
            value = patch[(kind, layer)](value)
        keep[(kind, layer)] = value
        return value

    h = point("resid", 0, tr.wte.weight[ids] + tr.wpe.weight[:T])
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    for layer, block in enumerate(tr.h):
        x = F.layer_norm(h, (c.d_model,), block.ln_1.weight, block.ln_1.bias, c.ln_eps)
        q, k, v = (x @ block.attn.c_attn.weight + block.attn.c_attn.bias).split(c.d_model, dim=-1)
        q, k, v = (t.view(B, T, c.heads, c.d_head).transpose(1, 2) for t in (q, k, v))  # (B, heads, T, d_head)
        pattern = (q @ k.transpose(-1, -2) / c.d_head**0.5).masked_fill(~causal, float("-inf")).softmax(-1)
        z = point("z", layer, (pattern @ v).transpose(1, 2).reshape(B, T, c.d_model))
        h = h + point("attn", layer, z @ block.attn.c_proj.weight + block.attn.c_proj.bias)
        x = F.layer_norm(h, (c.d_model,), block.ln_2.weight, block.ln_2.bias, c.ln_eps)
        mlp = F.gelu(x @ block.mlp.c_fc.weight + block.mlp.c_fc.bias, approximate="tanh") @ block.mlp.c_proj.weight
        h = h + point("mlp", layer, mlp + block.mlp.c_proj.bias)
        h = point("resid", layer + 1, h)
    last = F.layer_norm(h[:, -1], (c.d_model,), tr.ln_f.weight, tr.ln_f.bias, c.ln_eps)
    return last @ tr.wte.weight.T, keep


# %% [metric] Logit difference and the fraction recovered
def logit_diff(logits, io, s):
    """logits: (batch, vocab) at the last position → logit(indirect object) − logit(subject) per row."""
    return logits[:, io] - logits[:, s]


def recovered(patched, clean, corrupted):
    """0: the patch did nothing; 1: it brought the clean logit difference back; above 1 or below 0 also happen."""
    return (patched - corrupted) / (clean - corrupted)


# %% [resid] Patching the residual stream, one position at a time
def one_position(source):
    """Batch row b takes the clean value at position b and keeps the corrupted run's everywhere else."""
    def fn(value):
        value = value.clone()
        rows = torch.arange(value.shape[0])
        value[rows, rows] = source[0, rows]
        return value
    return fn


def patch_stream(model, clean_ids, bad_ids, io, s, kind="resid", layers=13):
    """(layers, T) logit differences: the corrupted prompt with the clean stream (or, with kind="attn" / "mlp", a
    layer's clean output) at one point and one position."""
    _, clean = run(model, clean_ids[None])
    T = clean_ids.shape[0]
    out = torch.zeros(layers, T)
    for layer in range(layers):
        logits, _ = run(model, bad_ids.expand(T, T), {(kind, layer): one_position(clean[(kind, layer)])})
        out[layer] = logit_diff(logits, io, s)
    return out


# %% [heads] Patching each head's output
def head_slices(source, heads, positions, c: Config = Config()):
    """Row b takes the clean output of heads[b] (its 64 columns of z) at positions[b] (an index or all of them)."""
    def fn(value):
        value = value.clone()
        for b, (head, pos) in enumerate(zip(heads, positions)):
            cols = slice(head * c.d_head, (head + 1) * c.d_head)
            value[b, pos, cols] = source[0, pos, cols]
        return value
    return fn


def patch_heads(model, clean_ids, bad_ids, io, s, by_position=False, c: Config = Config()):
    """(layers, heads) logit differences with one head's clean output at every position, or with by_position
    (layers, heads, T) with one head's clean output at one position."""
    _, clean = run(model, clean_ids[None])
    T = clean_ids.shape[0]
    out = torch.zeros(c.layers, c.heads, T) if by_position else torch.zeros(c.layers, c.heads)
    for layer in range(c.layers):
        if by_position:
            heads = [h for h in range(c.heads) for _ in range(T)]
            positions = [t for _ in range(c.heads) for t in range(T)]
        else:
            heads, positions = list(range(c.heads)), [slice(None)] * c.heads
        logits, _ = run(model, bad_ids.expand(len(heads), T), {("z", layer): head_slices(clean[("z", layer)], heads,
                                                                                          positions)})
        out[layer] = logit_diff(logits, io, s).view(out[layer].shape)
    return out


# %% [model] The whole sweep
def sweep(model, clean_ids, bad_ids, io, s):
    """Every patch of the tab for one prompt pair, with the clean and corrupted baselines."""
    clean = logit_diff(run(model, clean_ids[None])[0], io, s)[0]
    corrupted = logit_diff(run(model, bad_ids[None])[0], io, s)[0]
    return {
        "clean": clean, "corrupted": corrupted,
        "resid": patch_stream(model, clean_ids, bad_ids, io, s, "resid", 13),
        "attn": patch_stream(model, clean_ids, bad_ids, io, s, "attn", 12),
        "mlp": patch_stream(model, clean_ids, bad_ids, io, s, "mlp", 12),
        "heads": patch_heads(model, clean_ids, bad_ids, io, s),
        "headsByPosition": patch_heads(model, clean_ids, bad_ids, io, s, by_position=True),
    }


# model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
# clean = torch.tensor(tokenizer("When Mary and John went to the store, John gave a drink to").input_ids)
# bad = torch.tensor(tokenizer("When Mary and John went to the store, Mary gave a drink to").input_ids)
# out = sweep(model, clean, bad, io=tokenizer.encode(" Mary")[0], s=tokenizer.encode(" John")[0])
# recovered(out["heads"], out["clean"], out["corrupted"])  # (layers, heads); the IOI paper names 9.9, 9.6, 10.0
`;export{e as default};
