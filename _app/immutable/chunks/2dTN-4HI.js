const e=`# The tuned lens (Belrose et al. 2023) on GPT-2 small, the tuned-lens tab, as plain PyTorch.
# The logit lens reads every block's stream with the model's final readout, but early streams are not yet in the
# basis that readout expects. The tuned lens learns, for each point ℓ of the stream, one affine map that moves the
# stream there toward what the final stream would be, h ↦ h + A_ℓ h + b_ℓ, trained so that reading the moved stream
# matches the model's own output distribution (KL). The tab uses AlignmentResearch's released translators for GPT-2,
# lens/gpt2/params.pt in their Hugging Face Space; GPT-2's own final LayerNorm and unembedding do the reading.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    layers: int = 12  # one translator per point 0 … 11; the stream after block 12 is the output and needs none
    d_model: int = 768
    ln_eps: float = 1e-5


# %% [translator] One affine translator per point of the stream
def load_translators(params):
    """params: the released state dict, {"ℓ.weight": (768, 768), "ℓ.bias": (768,)} for ℓ = 0 … 11 → [(A_ℓ, b_ℓ)]."""
    return [(params[f"{layer}.weight"], params[f"{layer}.bias"]) for layer in range(len(params) // 2)]


def translate(h, translator):
    """h: (..., d_model) the stream at one point → the stream moved toward the final one. The map is stored as a
    residual on top of the identity (so its training's weight decay pulls it toward doing nothing)."""
    A, b = translator
    return h + h @ A.T + b


# %% [lens] Reading the translated stream
def unembed(model, h, c: Config = Config()):
    """GPT-2's own readout: final LayerNorm, then the tied unembedding (W_Eᵀ)."""
    ln = model.transformer.ln_f
    return F.layer_norm(h, (c.d_model,), ln.weight, ln.bias, c.ln_eps) @ model.transformer.wte.weight.T


def tuned_lens(model, translators, h, point):
    """h: (..., d_model) the stream at point 0 … 12 → (..., vocab). Point 12 is the model's own output."""
    return unembed(model, h if point == len(translators) else translate(h, translators[point]))


def logit_lens(model, h):
    """The same reading without the translator, for comparison."""
    return unembed(model, h)


# %% [kl] Distance from the model's output
def kl_to_final(lens_logits, final_logits):
    """KL(final ‖ lens) in nats, per position: what the tuned lens is trained to make small."""
    final = final_logits.double().log_softmax(-1)
    return (final.exp() * (final - lens_logits.double().log_softmax(-1))).sum(-1)


def readout(lens_logits, final_logits, target, k=5):
    """The lens's top-k tokens, the correct next token's probability and rank, and the KL to the final output."""
    logp = lens_logits.double().log_softmax(-1)
    top_p, top_ids = logp.exp().topk(k, dim=-1)
    target_logp = logp.gather(-1, target[..., None])[..., 0]
    return {"top_ids": top_ids, "top_p": top_p, "target_p": target_logp.exp(),
            "target_rank": (logp > target_logp[..., None]).sum(-1) + 1, "kl": kl_to_final(lens_logits, final_logits)}


# %% [corpus] Averages over a corpus
@torch.no_grad()
def corpus_means(model, translators, stream, final_logits, target):
    """stream: (13, T, d) at T positions of real text; final_logits: (T, vocab); target: (T,) next tokens → for both
    lenses, per point, the summed KL to the final output, cross-entropy of the next token and agreement of the top
    token with the model's (sums, so windows add up; divide by the number of positions)."""
    final = final_logits.double().log_softmax(-1)
    sums = {name: torch.zeros(stream.shape[0], dtype=torch.float64)
            for name in ("kl.logit", "kl.tuned", "nll.logit", "nll.tuned", "agree.logit", "agree.tuned")}
    for point in range(stream.shape[0]):
        for name, logits in (("logit", logit_lens(model, stream[point])),
                             ("tuned", tuned_lens(model, translators, stream[point], point))):
            logp = logits.double().log_softmax(-1)
            sums[f"kl.{name}"][point] = (final.exp() * (final - logp)).sum()
            sums[f"nll.{name}"][point] = -logp.gather(-1, target[:, None]).sum()
            sums[f"agree.{name}"][point] = (logp.argmax(-1) == final.argmax(-1)).double().sum()
    return sums


# %% [model] Both lenses on a prompt
@torch.no_grad()
def compare(model, translators, stream, ids):
    """stream: (13, T, d), the residual stream of a prompt (as in the logit-lens tab) → both lenses' readouts at every
    point of every position that has a next token in the text."""
    stream = stream[:, : len(ids) - 1]
    final = unembed(model, stream[-1])
    target = ids[1:]
    out = {}
    for name, read in (("logit", lambda h, p: logit_lens(model, h)),
                       ("tuned", lambda h, p: tuned_lens(model, translators, h, p))):
        logits = torch.stack([read(stream[p], p) for p in range(stream.shape[0])])
        out[name] = readout(logits, final.expand_as(logits), target.expand(stream.shape[0], -1))
    return out


# params = torch.load("lens/gpt2/params.pt", weights_only=True)  # AlignmentResearch/tuned-lens Space
# translators = load_translators(params)
# both = compare(model, translators, stream, ids)  # stream from the logit-lens tab's residual_stream
`;export{e as default};
