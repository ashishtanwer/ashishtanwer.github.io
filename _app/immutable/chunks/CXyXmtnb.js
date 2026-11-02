const e=`# A Gemma Scope transcoder in place of Gemma 2 2B's MLP 12, the transcoders tab, as plain PyTorch
# (Dunefsky et al. 2024, arXiv:2406.11944). An MLP reads the stream entering it and writes one vector back; a transcoder
# is trained to write the same vector through a wide, sparse layer of features: encode the MLP's input with a JumpReLU
# (each feature fires only above its own threshold), decode the few that fire. Swapping it in for the MLP shows how
# much of the MLP's job the features capture. Gemma Scope folds the pre-MLP RMSNorm's gain into the MLP, so the
# transcoder reads x / rms(x) without it, and it predicts the MLP's output after the post-MLP norm.
# Weights: google/gemma-2-2b; transcoders: google/gemma-scope-2b-pt-transcoders layer_12/width_16k.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    layer: int = 12
    d_model: int = 2304
    d_mlp: int = 9216
    features: int = 16384
    rms_eps: float = 1e-6
    final_softcap: float = 30.0


# %% [mlp] Gemma 2's MLP sublayer
def mlp_sublayer(layer, mid, c: Config = Config()):
    """mid (T, d): the stream after the block's attention → (x / rms(x), the transcoder's input; the MLP's output after
    its own norm, the term the block adds to the stream)."""
    x_norm = mid * torch.rsqrt(mid.pow(2).mean(-1, keepdim=True) + c.rms_eps)
    x = x_norm * (1 + layer.pre_feedforward_layernorm.weight)  # the gain the transcoder does not see
    m = layer.mlp
    y = (F.gelu(x @ m.gate_proj.weight.T, approximate="tanh") * (x @ m.up_proj.weight.T)) @ m.down_proj.weight.T
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + c.rms_eps) * (1 + layer.post_feedforward_layernorm.weight)
    return x_norm, y


# %% [encode] The JumpReLU encoder
def encode(tc, x_norm):
    """(T, d) → pre-activations and features (T, 16384): a feature keeps its pre-activation only above its threshold θ;
    below it, it is exactly 0, which is what makes the code sparse."""
    pre = x_norm @ tc["W_enc"] + tc["b_enc"]
    return pre, pre * (pre > tc["threshold"])


# %% [decode] The decoder
def decode(tc, acts):
    """ŷ = Σ_i f_i · W_dec[i] + b_dec: only the active features' rows add anything."""
    return acts @ tc["W_dec"] + tc["b_dec"]


# %% [error] What the transcoder misses
def error_stats(y, yhat):
    """Per token: ‖y‖, ‖y − ŷ‖ and cos(y, ŷ); and the fraction of y's variance (over tokens) left unexplained."""
    err = y - yhat
    fvu = err.pow(2).sum() / (y - y.mean(0)).pow(2).sum()
    return y.norm(dim=-1), err.norm(dim=-1), F.cosine_similarity(y, yhat, dim=-1), fvu


# %% [splice] The model with the transcoder in place of the MLP
@torch.no_grad()
def spliced_logits(model, ids, replace, c: Config = Config()):
    """Logits of ids (T,) with MLP \`c.layer\`'s output swapped by replace(x_norm, y) → new output (None: the model
    as it is), through hooks on the model's own norms. The other 25 MLPs are untouched."""
    seen = {}
    layer = model.model.layers[c.layer]

    def keep(module, args):  # the stream entering the MLP, normalised as the transcoder reads it
        seen["x_norm"] = args[0] * torch.rsqrt(args[0].pow(2).mean(-1, keepdim=True) + c.rms_eps)

    def swap(module, args, y):  # the MLP's output after its norm, replaced
        return y if replace is None else replace(seen["x_norm"], y)

    handles = [layer.pre_feedforward_layernorm.register_forward_pre_hook(keep),
               layer.post_feedforward_layernorm.register_forward_hook(swap)]
    try:
        return model(ids[None]).logits[0]  # Gemma 2 softcaps its logits at 30
    finally:
        for handle in handles:
            handle.remove()


def next_token_loss(logits, ids):
    """−log p(next token) at positions 0 … T − 2."""
    return -logits[:-1].log_softmax(-1).gather(-1, ids[1:, None])[:, 0]


# %% [contexts] Where a feature fires on text
@torch.no_grad()
def feature_on_text(model, tc, feature, ids, c: Config = Config()):
    """The feature's activation at every token of ids (T,), <bos> left out: run the model to MLP c.layer's input."""
    seen = {}
    layer = model.model.layers[c.layer]
    handle = layer.pre_feedforward_layernorm.register_forward_pre_hook(lambda m, a: seen.setdefault("mid", a[0]))
    try:
        model(ids[None])
    finally:
        handle.remove()
    x_norm, _ = mlp_sublayer(layer, seen["mid"][0], c)
    _, acts = encode(tc, x_norm)
    acts[0] = 0
    return acts[:, feature]


# %% [logits] A feature's decoder row through the unembedding
def logit_effect(model, tc, feature):
    """Which tokens the feature's write pushes up or down, read directly: W_U · (W_dec[f] ⊙ (1 + w_final))."""
    return model.lm_head.weight @ (tc["W_dec"][feature] * (1 + model.model.norm.weight))


# %% [model] One prompt through the transcoder
@torch.no_grad()
def transcoder_run(model, ids, tc, c: Config = Config()):
    """The MLP's real output y at layer c.layer, the transcoder's features and ŷ, and the next-token loss with the MLP,
    with ŷ in its place, and with the MLP removed."""
    seen = {}
    layer = model.model.layers[c.layer]
    handle = layer.pre_feedforward_layernorm.register_forward_pre_hook(lambda m, a: seen.setdefault("mid", a[0]))
    try:
        model(ids[None])
    finally:
        handle.remove()
    x_norm, y = mlp_sublayer(layer, seen["mid"][0], c)
    pre, acts = encode(tc, x_norm)
    yhat = decode(tc, acts)
    losses = [next_token_loss(spliced_logits(model, ids, r, c), ids)
              for r in (None, lambda x, y_: decode(tc, encode(tc, x)[1]), lambda x, y_: torch.zeros_like(y_))]
    return y, acts, yhat, losses
`;export{e as default};
