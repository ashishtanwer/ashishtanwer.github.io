const e=`# A Gemma Scope sparse autoencoder (Lieberum et al. 2024, arXiv:2408.05147) on Gemma 2 2B's residual stream: the
# 2304-dimensional output of block 12 is rewritten as a sparse, non-negative mix of 16,384 learned feature directions.
# The encoder decides which features are on (JumpReLU: each feature has its own threshold), the decoder adds their
# directions back up, and whatever it misses is the reconstruction error.
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    hidden: int = 2304
    features: int = 16384
    layer: int = 12  # counting blocks from 0: the SAE reads the output of the 13th of 26 blocks
    sae_file: str = "layer_12/width_16k/average_l0_82/params.npz"  # in google/gemma-scope-2b-pt-res


def load_sae(path):
    with np.load(path) as npz:  # W_enc (2304, 16384), b_enc, W_dec (16384, 2304), b_dec, threshold (16384)
        return {name: torch.from_numpy(npz[name]).float() for name in npz.files}


# %% [residual] The residual stream after block 12
class Stop(Exception):
    pass


def residual_after(model, ids, layer=12):
    """(batch, tokens, 2304): block \`layer\`'s output. The forward stops there; nothing later runs."""
    seen = []

    def grab(module, args, output):
        seen.append(output[0] if isinstance(output, tuple) else output)
        raise Stop

    handle = model.model.layers[layer].register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(ids)
    except Stop:
        pass
    finally:
        handle.remove()
    return seen[0]


# %% [encode] JumpReLU encoder: a threshold per feature
def encode(sae, x):
    """Pre-activations x·W_enc + b_enc for all 16,384 features; a feature is on only above its own threshold, and
    then it keeps its whole pre-activation (a jump from 0 to θ, not a ramp from 0 like ReLU)."""
    pre = x @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


# %% [decode] Decoder: features back to a residual vector
def decode(sae, features):
    """x̂ = Σ_i f_i · W_dec[i] + b_dec: each active feature adds its (unit-length) decoder direction, scaled."""
    return features @ sae["W_dec"] + sae["b_dec"]


# %% [error] How much the reconstruction misses
def reconstruction_error(x, x_hat):
    """Per token: the error's length, that length relative to ‖x‖, and the angle between x and x̂."""
    error_norm = (x - x_hat).norm(dim=-1)
    return {
        "error_norm": error_norm,
        "relative": error_norm / x.norm(dim=-1),
        "cosine": torch.nn.functional.cosine_similarity(x, x_hat, dim=-1),
    }


# %% [splice] Running the model on the reconstruction
def spliced_logits(model, ids, x_hat, layer=12):
    """Logits with block \`layer\`'s output replaced by x̂ at every position but the first (<bos>, which the SAE was not
    trained on): what the rest of the model makes of the SAE's version of the residual."""
    def swap(module, args, output):
        h = (output[0] if isinstance(output, tuple) else output).clone()
        h[:, 1:] = x_hat[:, 1:]
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(swap)
    try:
        with torch.no_grad():
            return model(ids).logits
    finally:
        handle.remove()


def next_token_loss(logits, ids):
    """−log p(the real next token) at every position but the last."""
    return -logits[:, :-1].double().log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]


# %% [contexts] Where a feature fires on real text
def top_contexts(model, sae, sample_ids, features, k=8, layer=12, batch=16):
    """sample_ids: (paragraphs, tokens), each starting with <bos>; features: a list of feature ids. For each feature,
    the k paragraphs where it peaks highest, the token it peaks on, and its activation on every token of them.
    <bos> is left out."""
    acts = []
    for start in range(0, len(sample_ids), batch):
        x = residual_after(model, sample_ids[start:start + batch], layer)
        acts.append(encode(sae, x)[..., features].permute(2, 0, 1))  # (features, batch, tokens)
    acts = torch.cat(acts, dim=1)
    acts[:, :, 0] = 0
    peak, where = acts.max(-1)  # (features, paragraphs)
    best = peak.topk(k, dim=-1).indices  # (features, k)
    return best, where.gather(-1, best), acts.gather(1, best[..., None].expand(-1, -1, acts.shape[-1]))


# %% [logits] What a feature's direction says to the unembedding
def logit_effect(model, sae, feature):
    """The decoder direction through the final norm's per-dimension scale and the unembedding, skipping blocks 13–25:
    which tokens this feature pushes up or down directly."""
    scale = 1 + model.model.norm.weight  # Gemma's RMSNorm multiplies by (1 + w)
    return model.lm_head.weight @ (sae["W_dec"][feature] * scale)


# %% [model] One prompt through the SAE
def sae_on_prompt(model, tokenizer, sae, text, layer=12):
    ids = torch.tensor([tokenizer(text).input_ids])  # starts with <bos>
    x = residual_after(model, ids, layer)[0]
    features = encode(sae, x)
    x_hat = decode(sae, features)
    stats = {"l0": (features > 0).sum(-1), **reconstruction_error(x, x_hat)}
    with torch.no_grad():
        clean = next_token_loss(model(ids).logits, ids)
    spliced = next_token_loss(spliced_logits(model, ids, x_hat[None], layer), ids)
    return features, x_hat, stats, clean, spliced
`;export{e as default};
