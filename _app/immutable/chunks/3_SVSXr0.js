const e=`# Induction heads in GPT-2 small, the induction-heads tab, as plain PyTorch.
# Random tokens written twice: nothing in the first copy can be predicted, but every token of the second copy can be
# read off the first. An induction head does it: at the current token it attends to the token that followed the same
# token's earlier occurrence (the "induction stripe", N − 1 positions back) and copies it into the stream. It finds
# that token through a previous-token head in an earlier layer, which wrote "the token before me" into each position.
# Weights: openai-community/gpt2 (GPT2LMHeadModel); scores and ablations as Olsson et al. 2022 (arXiv:2209.11895).
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
    vocab: int = 50257
    repeat: int = 25  # tokens per copy
    sequences: int = 32
    seed: int = 0


# %% [repeat] Random tokens, said twice
def repeated_tokens(c: Config = Config()):
    """(sequences, 2·repeat + 1): <|endoftext|>, \`repeat\` uniform random ids (never <|endoftext|>), the same again."""
    eot = c.vocab - 1
    g = torch.Generator().manual_seed(c.seed)
    r = torch.randint(0, eot, (c.sequences, c.repeat), generator=g)
    return torch.cat([torch.full((c.sequences, 1), eot), r, r], 1)


# %% [attention] Every head's attention pattern
def forward(model, ids, ablate=None, c: Config = Config()):
    """ids: (B, T) → logits (B, T, vocab), patterns (B, layers, heads, T, T) and each layer's head outputs z (B, T, d,
    heads side by side, before W_O). \`ablate\` maps layer → fn(z) → z, applied before W_O (a head's ablation)."""
    tr = model.transformer
    B, T = ids.shape
    h = tr.wte.weight[ids] + tr.wpe.weight[:T]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    patterns, zs = [], []
    for layer, block in enumerate(tr.h):
        x = F.layer_norm(h, (c.d_model,), block.ln_1.weight, block.ln_1.bias, c.ln_eps)
        q, k, v = (x @ block.attn.c_attn.weight + block.attn.c_attn.bias).split(c.d_model, dim=-1)
        q, k, v = (t.view(B, T, c.heads, c.d_head).transpose(1, 2) for t in (q, k, v))
        pattern = (q @ k.transpose(-1, -2) / c.d_head**0.5).masked_fill(~causal, float("-inf")).softmax(-1)
        z = (pattern @ v).transpose(1, 2).reshape(B, T, c.d_model)
        if ablate and layer in ablate:
            z = ablate[layer](z)
        h = h + z @ block.attn.c_proj.weight + block.attn.c_proj.bias
        x = F.layer_norm(h, (c.d_model,), block.ln_2.weight, block.ln_2.bias, c.ln_eps)
        mlp = F.gelu(x @ block.mlp.c_fc.weight + block.mlp.c_fc.bias, approximate="tanh") @ block.mlp.c_proj.weight
        h = h + mlp + block.mlp.c_proj.bias
        patterns.append(pattern)
        zs.append(z)
    ln = tr.ln_f
    logits = F.layer_norm(h, (c.d_model,), ln.weight, ln.bias, c.ln_eps) @ tr.wte.weight.T
    return logits, torch.stack(patterns, 1), zs, h


# %% [scores] Previous-token, duplicate and induction scores
def head_scores(patterns, c: Config = Config()):
    """patterns (B, layers, heads, T, T) → per sequence and head, the mean weight a query puts on the previous token,
    on its own earlier copy (p − repeat) and on the token after that copy (p − repeat + 1): the induction score."""
    N, T = c.repeat, patterns.shape[-1]
    p = torch.arange(1, T)
    q = torch.arange(N + 1, T)  # the second copy
    return {
        "previous": patterns[..., p, p - 1].mean(-1),
        "duplicate": patterns[..., q, q - N].mean(-1),
        "induction": patterns[..., q, q - N + 1].mean(-1),
    }


# %% [copy] Each head's push on the right next token
def copy_shares(model, ids, zs, final, c: Config = Config()):
    """Each head's direct share of logit(next token) − mean logit at positions 0 … T − 2, (B, layers, heads, T − 1):
    its output z_h W_O,h read through the final LayerNorm with the scale σ of this position frozen."""
    tr = model.transformer
    W_E = tr.wte.weight
    sigma = (final[:, :-1].var(-1, unbiased=False, keepdim=True) + c.ln_eps).sqrt()  # (B, T − 1, 1)
    direction = W_E[ids[:, 1:]] - W_E.mean(0)  # (B, T − 1, d)
    out = []
    for layer, block in enumerate(tr.h):
        B, T, _ = zs[layer].shape
        z = zs[layer][:, :-1].view(B, T - 1, c.heads, c.d_head)
        per_head = torch.einsum("bthd,hdm->bthm", z, block.attn.c_proj.weight.view(c.heads, c.d_head, c.d_model))
        centred = (per_head - per_head.mean(-1, keepdim=True)) * tr.ln_f.weight / sigma[:, :, None]
        out.append((centred * direction[:, :, None]).sum(-1).transpose(1, 2))  # (B, heads, T − 1)
    return torch.stack(out, 1)


# %% [ablate] Mean-ablating a head
def mean_ablation(heads, z_mean, c: Config = Config()):
    """heads: [(layer, head)] → \`ablate\` for forward(): each listed head's z replaced at positions 1 … T − 1 by its
    mean z_mean[layer] (over sequences and those positions), so the head writes the same thing everywhere."""
    by_layer = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)

    def make(layer, hs):
        def fn(z):
            z = z.clone()
            for h in hs:
                z[:, 1:, h * c.d_head:(h + 1) * c.d_head] = z_mean[layer][h * c.d_head:(h + 1) * c.d_head]
            return z
        return fn

    return {layer: make(layer, hs) for layer, hs in by_layer.items()}


# %% [model] The loss on both copies
def next_token_loss(logits, ids):
    """−log p(next token) at positions 0 … T − 2: the first copy (1 … repeat) is guesswork, the second is copying."""
    return -logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]


@torch.no_grad()
def induction_run(model, induction_heads, c: Config = Config()):
    ids = repeated_tokens(c)
    logits, patterns, zs, final = forward(model, ids)
    z_mean = {layer: z[:, 1:].mean((0, 1)) for layer, z in enumerate(zs)}
    ablated, _, _, _ = forward(model, ids, ablate=mean_ablation(induction_heads, z_mean))
    return head_scores(patterns), next_token_loss(logits, ids), next_token_loss(ablated, ids)


# model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
# scores, clean, ablated = induction_run(model, [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)])
# clean[:, 26:].mean(), ablated[:, 26:].mean()  # the second copy, before and after
`;export{e as default};
