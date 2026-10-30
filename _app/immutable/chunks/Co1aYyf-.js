const e=`# Self-consistency (Wang et al. 2022) on Qwen2.5-Math-1.5B-Instruct and GSM8K, as plain PyTorch.
# Instead of one greedy chain of thought, sample many at a temperature, read each one's final answer, and answer
# with the most common one: different reasoning paths that reach the same number reinforce each other, while
# mistakes tend to scatter. Every answer is drawn from its own seeded stream of uniforms, so a token reproduces from
# its logits and the seed. Each "# %% [id]" section is one card of the tab.
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    samples: int = 32  # N answers per problem
    temperature: float = 0.7  # the self-consistency paper's setting for PaLM
    top_k: int = 40
    max_new_tokens: int = 512  # the model card's example
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"
    system: str = "Please reason step by step, and put your final answer within \\\\boxed{}."


# %% [sampling] Sampling one answer token
def sampling_probs(logits, temperature, top_k):
    """logits (..., V) → the distribution the next token is drawn from: divide by the temperature, keep the top_k."""
    scores = logits / temperature
    kth = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
    return scores.masked_fill(scores < kth, float("-inf")).softmax(-1)


def draw(probs, u):
    """Inverse CDF: lay the tokens out on [0, 1) from the most likely down (ties in id order); the token whose slice
    contains u is drawn. Greedy decoding is the token at the bottom of [0, 1): the most likely one."""
    ordered, order = probs.sort(dim=-1, descending=True, stable=True)
    j = (ordered.cumsum(-1) <= u[..., None]).sum(-1)
    j = torch.minimum(j, (ordered > 0).sum(-1) - 1)
    return order.gather(-1, j[..., None])[..., 0]


def uniforms(seed, length):
    """One answer's stream: its t-th token uses u[t]. Answer j of GSM8K row r is seeded 1000·r + j."""
    return torch.rand(length, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# %% [answer] Extracting the final answer
def last_boxed(text):
    """Content of the last \\\\boxed{…} (braces may nest), or None."""
    start = text.rfind("\\\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len("\\\\boxed")
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1 : j]
    return None


def as_number(answer):
    """A boxed answer as a number: units in \\\\text{…}, $, %, commas and spaces dropped; fractions divided."""
    if answer is None:
        return None
    s = re.sub(r"\\\\(?:text|mbox|mathrm)\\{[^{}]*\\}", "", answer)
    s = re.sub(r"\\\\[dt]?frac\\{([^{}]*)\\}\\{([^{}]*)\\}", r"(\\1)/(\\2)", s)
    for junk in ("\\\\$", "$", "\\\\%", "%", "\\\\!", "\\\\,", ",", " ", "{", "}"):
        s = s.replace(junk, "")
    m = re.fullmatch(r"\\(?(-?\\d+(?:\\.\\d+)?)\\)?(?:/\\(?(-?\\d+(?:\\.\\d+)?)\\)?)?", s.rstrip("."))
    if not m or (m[2] is not None and float(m[2]) == 0):
        return None
    return float(m[1]) / (float(m[2]) if m[2] is not None else 1.0)


def is_correct(value, gold):
    return value is not None and abs(value - float(gold)) < 1e-6


# %% [vote] Majority vote
def majority(values):
    """values: each answer's number (None if it has none) → (the answer with the most votes, {answer: votes}).
    Answers without a number do not vote; a tie goes to the answer voted for first."""
    counts, first = {}, {}
    for i, v in enumerate(values):
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
            first.setdefault(v, i)
    if not counts:
        return None, counts
    return max(counts, key=lambda v: (counts[v], -first[v])), counts


# %% [accuracy] Accuracy against the number of samples
def vote_accuracy(values, golds, n, orderings=256, seed=0):
    """Accuracy of a majority vote over n of each problem's answers, averaged over seeded random orderings of the
    answers (each ordering's first n vote), and the coverage: how often any of the n is right."""
    generator = torch.Generator().manual_seed(seed)
    perms = [torch.randperm(len(values[0]), generator=generator).tolist() for _ in range(orderings)]
    hits, covered = 0.0, 0.0
    for answers, gold in zip(values, golds):
        for perm in perms:
            chosen = [answers[i] for i in perm[:n]]
            hits += is_correct(majority(chosen)[0], gold)
            covered += any(is_correct(v, gold) for v in chosen)
    total = len(values) * len(perms)
    return torch.tensor(hits / total, dtype=torch.float64), torch.tensor(covered / total, dtype=torch.float64)


# %% [model] Self-consistency
@torch.no_grad()
def sample_answer(model, prompt_ids, u, c: Config = Config(), greedy=False):
    """One answer, token by token: greedy takes the most likely token, sampling uses u[t] for token t. No cache."""
    ids, out = list(prompt_ids), []
    for t in range(c.max_new_tokens):
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1].double()
        token = int(logits.argmax()) if greedy else int(draw(sampling_probs(logits, c.temperature, c.top_k), u[t]))
        ids.append(token)
        out.append(token)
        if token in c.stops:
            break
    return out


def self_consistency(model, tokenizer, prompt_ids, row, c: Config = Config()):
    """N sampled answers to one problem, their final answers and the majority vote."""
    answers = [sample_answer(model, prompt_ids, uniforms(1000 * row + j, c.max_new_tokens + 8), c)
               for j in range(c.samples)]
    values = [as_number(last_boxed(tokenizer.decode(a, skip_special_tokens=True))) for a in answers]
    return answers, values, majority(values)[0]


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", dtype=torch.float32)
# prompt = tokenizer.apply_chat_template([{"role": "system", "content": Config.system},
#                                         {"role": "user", "content": question}], add_generation_prompt=True)
# answers, values, voted = self_consistency(model, tokenizer, prompt, row=gsm8k_row)
`;export{e as default};
