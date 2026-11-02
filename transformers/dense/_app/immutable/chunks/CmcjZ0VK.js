const e=`# SynthID-Text (Dathathri et al. 2024) on Qwen2.5-0.5B-Instruct, as plain PyTorch, following transformers'
# SynthIDTextWatermarkLogitsProcessor. Every candidate token gets one random bit per layer, a g-value, from a hash of
# the last 4 tokens, the candidate and that layer's key. Tournament sampling then favours candidates with more ones:
# in each layer two candidates drawn from the distribution play, the larger g-value wins. On average over keys the
# distribution is unchanged, but the written text carries more ones than chance, which a detector with the keys can
# count. Each "# %% [id]" section is one card of the tab.
import math
from dataclasses import dataclass

import torch

KEYS = (654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468, 990, 966, 226, 324, 585, 118, 504,
        421, 521, 129, 669, 732, 225, 90, 960)  # transformers' SynthID example
MULTIPLIER, INCREMENT = 6364136223846793005, 1  # the hash's linear congruential step


@dataclass
class Config:
    keys: tuple = KEYS  # one per tournament layer: 30 layers
    ngram_len: int = 5  # the candidate and the 4 tokens before it
    table_size: int = 2**16
    table_seed: int = 0
    top_k: int = 40
    temperature: float = 1.0
    max_new_tokens: int = 200
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"
    eos: int = 151645  # the detector ignores everything from the first "<|im_end|>" on


# %% [g-values] The g-values
def sampling_table(c: Config = Config()):
    """The random bits every hash is looked up in, drawn once from the table seed (on the CPU: another device's
    generator draws another table)."""
    return torch.randint(0, 2, (c.table_size,), generator=torch.Generator().manual_seed(c.table_seed))


def accumulate(h, values):
    """For each value: h ← (h + value) · MULTIPLIER + INCREMENT, wrapping in int64. values (..., n)."""
    for i in range(values.shape[-1]):
        h = (h + values[..., i]) * MULTIPLIER + INCREMENT
    return h


def g_values(context, candidates, table, c: Config = Config()):
    """context (4,) and candidates (n,) token ids → (n, layers) g-values: hash the context from 1, then each
    candidate, then each layer's key, and read the table at the hash mod its size."""
    h = accumulate(torch.ones((), dtype=torch.long), torch.as_tensor(context, dtype=torch.long))
    h = accumulate(h, torch.as_tensor(candidates, dtype=torch.long)[:, None])  # (n,)
    keys = torch.tensor(c.keys, dtype=torch.long)
    h = accumulate(h[:, None], keys[None, :, None])  # (n, layers)
    return table[h % c.table_size]


# %% [tournament] Tournament sampling
def tournament(probs, g):
    """probs (n,) over the kept candidates, g (n, layers) → the distribution of the tournament's winner. In each layer
    two candidates are drawn from the current distribution and the one with the larger g-value wins (a tie: either),
    so a candidate's chance becomes p · (1 + g − E_p[g])."""
    for layer in range(g.shape[-1]):
        probs = probs * (1 + g[:, layer] - (g[:, layer] * probs).sum())
    return probs


# %% [sampling] Sampling one token
def watermark(logits, state, table, c: Config = Config()):
    """logits (vocab,) → the kept ids (top_k,) and their probabilities. state: the last 4 written tokens (zeros at the
    start) and the contexts seen so far. A context seen before is left unwatermarked: the same g-values would push the
    text the same way twice, and the detector skips those tokens too."""
    values, ids = logits.topk(c.top_k)
    probs = (values / c.temperature).double().softmax(-1)
    seen = tuple(state["context"])
    if seen in state["seen"]:
        return ids, probs
    state["seen"].add(seen)
    return ids, tournament(probs, g_values(state["context"], ids, table, c).double())


def uniforms(seed, length):
    """One answer's stream of uniforms: its t-th token uses u[t] (the Watermark tab's seeds and streams)."""
    return torch.rand(length, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def draw(probs, u):
    """Inverse CDF: lay the tokens out on [0, 1) from the most likely down (ties in index order); the one whose
    slice holds u is drawn."""
    ordered, order = probs.sort(dim=-1, descending=True, stable=True)
    j = (ordered.cumsum(-1) <= u[..., None]).sum(-1)
    j = torch.minimum(j, (ordered > 0).sum(-1) - 1)
    return order.gather(-1, j[..., None])[..., 0]


# %% [detect] The detector
def score(ids, table, c: Config = Config()):
    """A text's token ids (no prompt) → (ones, trials, mean g-value, p-value). Tokens from the 5th on are scored with
    the 4 tokens before them, unless that context appeared earlier in the text or the token is at or after the first
    "<|im_end|>". Without the keys every g-value is a fair coin, so the p-value is P(Binomial(trials, ½) ≥ ones)."""
    seen, ones, scored = set(), 0, 0
    n = c.ngram_len - 1
    for t in range(n, len(ids)):
        if c.eos in ids[: t + 1]:
            break
        context = tuple(ids[t - n : t])
        if context in seen:
            continue
        seen.add(context)
        ones += int(g_values(context, [ids[t]], table, c).sum())
        scored += 1
    trials = scored * len(c.keys)
    term = total = math.comb(trials, ones)
    for k in range(ones, trials):  # the tail's terms, C(n, k + 1) = C(n, k) · (n − k) / (k + 1)
        term = term * (trials - k) // (k + 1)
        total += term
    return ones, trials, ones / max(1, trials), total / 2**trials if trials else 1.0


# %% [model] Generating a watermarked answer
@torch.no_grad()
def generate(model, prompt_ids, u, c: Config = Config()):
    """One answer, token by token (no cache): watermark the kept distribution, then draw with u[t]."""
    table = sampling_table(c)
    state = {"context": [0] * (c.ngram_len - 1), "seen": set()}
    ids, out = list(prompt_ids), []
    for t in range(c.max_new_tokens):
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        kept, probs = watermark(logits, state, table, c)
        token = int(kept[draw(probs, u[t])])
        state["context"] = state["context"][1:] + [token]
        ids.append(token)
        out.append(token)
        if token in c.stops:
            break
    return out


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
# prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True)
# answer = generate(model, prompt, uniforms(mt_bench_id, 208))
# ones, trials, mean, p = score(answer, sampling_table())  # mean > ½ and a tiny p: watermarked
`;export{e as default};
