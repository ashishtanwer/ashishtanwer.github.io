const e=`# The logit lens and the tuned lens set against each other on GPT-2 small, the Lenses Compare tab, as plain PyTorch.
# Both read the residual stream at every point with GPT-2's own final LayerNorm and unembedding; the tuned lens first
# moves the stream with one learned affine map per point, h ↦ h + A_ℓ h + b_ℓ (AlignmentResearch's released
# translators for GPT-2, lens/gpt2/params.pt in their Hugging Face Space). The page asks how far each reading is from
# the model's output, where the model's own final favourite ranks in it, and how far the two readings are from each
# other, on the logit-lens tab's prompts and on WikiText-2.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    layers: int = 12  # translators for points 0 … 11; point 12, the stream after the last block, is the output
    d_model: int = 768
    ln_eps: float = 1e-5


# %% [lens] Two readings of one stream
def unembed(model, h, c: Config = Config()):
    """GPT-2's own readout: final LayerNorm, then the tied unembedding (W_Eᵀ)."""
    ln = model.transformer.ln_f
    return F.layer_norm(h, (c.d_model,), ln.weight, ln.bias, c.ln_eps) @ model.transformer.wte.weight.T


def load_translators(params):
    """params: the released state dict, {"ℓ.weight": (768, 768), "ℓ.bias": (768,)} for ℓ = 0 … 11 → [(A_ℓ, b_ℓ)]."""
    return [(params[f"{layer}.weight"], params[f"{layer}.bias"]) for layer in range(len(params) // 2)]


def read(model, translators, h, point, lens):
    """h: (..., d_model) the stream at point 0 … 12 → (..., vocab) logits. The logit lens reads h as it is; the tuned
    lens moves it first (stored on top of the identity: h + A h + b). At point 12 both are the model's output."""
    if lens == "tuned" and point < len(translators):
        A, b = translators[point]
        h = h + h @ A.T + b
    return unembed(model, h)


# %% [kl] Distance from the model's output
def kl_to_final(lens_logits, final_logits):
    """KL(final ‖ lens) in nats, per position: what the tuned lens is trained to make small."""
    final = final_logits.double().log_softmax(-1)
    return (final.exp() * (final - lens_logits.double().log_softmax(-1))).sum(-1)


# %% [rank] Rank of the model's own favourite
def favourite_rank(lens_logits, final_logits):
    """Where the token the model finally predicts stands in a lens's reading: its rank (1 = the lens's favourite too;
    ties count in its favour) and the probability the lens gives it. The members rank the text's next token instead,
    which the model does not always predict."""
    favourite = final_logits.argmax(-1, keepdim=True)
    logp = lens_logits.double().log_softmax(-1)
    at = logp.gather(-1, favourite)
    return (logp > at).sum(-1) + 1, at[..., 0].exp()


# %% [disagree] How far apart the two readings are
def js_divergence(logits_a, logits_b):
    """JS(p ‖ q) = ½ KL(p ‖ m) + ½ KL(q ‖ m), m = ½ (p + q), in nats: 0 when the readings agree, at most ln 2 when
    they share no probability. Symmetric, so it measures the two lenses against each other, not against the output.
    Computed in log space, so a token one lens gives almost nothing still has a finite log m."""
    logp, logq = logits_a.double().log_softmax(-1), logits_b.double().log_softmax(-1)
    logm = torch.logaddexp(logp, logq) - torch.log(torch.tensor(2.0, dtype=torch.float64))
    return 0.5 * (logp.exp() * (logp - logm)).sum(-1) + 0.5 * (logq.exp() * (logq - logm)).sum(-1)


# %% [corpus] Over a corpus
@torch.no_grad()
def corpus_summary(model, translators, windows, quartiles=(0.25, 0.5, 0.75)):
    """windows: one (stream (13, T, d), final_logits (T, vocab)) per stretch of real text → per point, over every
    position: the quartiles of the rank of the model's favourite under each lens (ranks that occur), the mean
    divergence between the lenses and how often their favourites agree."""
    points = Config().layers + 1
    ranks = {lens: [[] for _ in range(points)] for lens in ("logit", "tuned")}
    js, agree, count = torch.zeros(points, dtype=torch.float64), torch.zeros(points, dtype=torch.float64), 0
    for stream, final_logits in windows:
        for point in range(points):
            logits = {lens: read(model, translators, stream[point], point, lens) for lens in ("logit", "tuned")}
            for lens in ranks:
                ranks[lens][point].append(favourite_rank(logits[lens], final_logits)[0])
            js[point] += js_divergence(logits["logit"], logits["tuned"]).sum()
            agree[point] += (logits["logit"].argmax(-1) == logits["tuned"].argmax(-1)).double().sum()
        count += final_logits.shape[0]
    q = torch.tensor(quartiles, dtype=torch.float64)
    out = {f"rank.{lens}": torch.stack([torch.quantile(torch.cat(r).double(), q, interpolation="lower") for r in rows])
           for lens, rows in ranks.items()}
    return {**out, "js": js / count, "agree.between": agree / count}


# %% [model] Both lenses on a prompt
@torch.no_grad()
def compare(model, translators, stream):
    """stream: (13, T, d), the residual stream of a prompt (the logit-lens tab's residual_stream) → for both lenses,
    at every point and position: the KL from the output, the rank and probability of the output's favourite; and the
    divergence between the two readings."""
    final = unembed(model, stream[-1])
    points = range(stream.shape[0])
    logits = {lens: torch.stack([read(model, translators, stream[p], p, lens) for p in points])
              for lens in ("logit", "tuned")}
    out = {"js": js_divergence(logits["logit"], logits["tuned"])}
    for lens, values in logits.items():
        rank, p = favourite_rank(values, final.expand_as(values))
        out[lens] = {"kl": kl_to_final(values, final.expand_as(values)), "final_rank": rank, "final_p": p}
    return out


# params = torch.load("lens/gpt2/params.pt", weights_only=True)  # AlignmentResearch/tuned-lens Space
# translators = load_translators(params)
# both = compare(model, translators, stream)  # stream from the logit-lens tab's residual_stream
`;export{e as default};
