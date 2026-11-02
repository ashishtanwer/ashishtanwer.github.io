const e=`# The green-list watermark (Kirchenbauer et al. 2023) on Qwen2.5-0.5B-Instruct, as plain PyTorch.
# Before each token, a hash of the previous token seeds a random split of the vocabulary into a green list (a share
# γ of it) and a red list, and δ is added to the green tokens' logits, so the text leans green. Anyone with the key can
# count the green tokens of a text and test whether there are more than the γ·T that chance would give. Each
# "# %% [id]" section is one card of the tab.
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    vocab: int = 151936  # Qwen2.5's logits (its tokenizer uses 151,665 of them)
    gamma: float = 0.25  # the green share of the vocabulary
    delta: float = 2.0  # added to the green tokens' logits
    key: int = 15485863  # the authors' hash key
    top_k: int = 40
    temperature: float = 1.0
    max_new_tokens: int = 200
    z_threshold: float = 4.0
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"


# %% [greenlist] The green list
def green_list(prev, c: Config = Config()):
    """bool (vocab,), True for the green tokens after token \`prev\`: seed a generator with key × prev, permute the
    vocabulary and keep the first γ·|V| ids (the "left hash" of the authors' code and of transformers)."""
    generator = torch.Generator().manual_seed(c.key * prev % (2**64 - 1))
    green = torch.zeros(c.vocab, dtype=torch.bool)
    green[torch.randperm(c.vocab, generator=generator)[: int(c.vocab * c.gamma)]] = True
    return green


# %% [bias] Biasing the logits
def watermark(logits, prev, c: Config = Config()):
    """logits (vocab,) → the kept ids (top_k,) and their scores: keep the top_k logits, add δ to the green ones (after
    top-k, where transformers' generate applies its watermark processors), divide by the temperature."""
    values, ids = logits.topk(c.top_k)
    return ids, (values + c.delta * green_list(prev, c)[ids]) / c.temperature


# %% [sampling] Sampling one token
def uniforms(seed, length):
    """One answer's stream of uniforms: its t-th token uses u[t]. Answer p is seeded with its MT-Bench id, in every
    setting, so the answers with and without the watermark share their draws."""
    return torch.rand(length, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def draw(probs, u):
    """Inverse CDF: lay the tokens out on [0, 1) from the most likely down (ties in index order); the one whose
    slice holds u is drawn."""
    ordered, order = probs.sort(dim=-1, descending=True, stable=True)
    j = (ordered.cumsum(-1) <= u[..., None]).sum(-1)
    j = torch.minimum(j, (ordered > 0).sum(-1) - 1)
    return order.gather(-1, j[..., None])[..., 0]


# %% [detect] Detection: the z-score
def detect(ids, c: Config = Config()):
    """A text's token ids (no prompt) → (green tokens G, scored tokens T, z). Token t ≥ 1 is green when it is in the
    green list of token t − 1; without the watermark G ~ Binomial(T, γ), so z = (G − γT) / √(Tγ(1 − γ)) is about
    standard normal and z > 4 almost never happens by chance."""
    green = sum(int(green_list(ids[t - 1], c)[ids[t]]) for t in range(1, len(ids)))
    scored = len(ids) - 1
    return green, scored, (green - c.gamma * scored) / math.sqrt(scored * c.gamma * (1 - c.gamma))


# %% [robustness] Replacing tokens
def replace(ids, positions, alternatives):
    """The text with the tokens at \`positions\` swapped for \`alternatives[t]\` (the tab uses the model's most likely
    other token at that step). A replaced token changes two scores: its own, and the next token's, whose green list it
    seeds."""
    out = list(ids)
    for t in positions:
        out[t] = alternatives[t]
    return out


# %% [model] Generating a watermarked answer
@torch.no_grad()
def generate(model, prompt_ids, u, c: Config = Config()):
    """One answer, token by token (no cache): watermark the logits, then draw with u[t]."""
    ids, out = list(prompt_ids), []
    for t in range(c.max_new_tokens):
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        kept, scores = watermark(logits, ids[-1], c)
        token = int(kept[draw(scores.double().softmax(-1), u[t])])
        ids.append(token)
        out.append(token)
        if token in c.stops:
            break
    return out


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
# prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True)
# answer = generate(model, prompt, uniforms(mt_bench_id, 208))
# green, scored, z = detect(answer[:-1])  # without the stop token; z > 4: watermarked
`;export{e as default};
