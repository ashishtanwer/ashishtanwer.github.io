const n=`# Thinking mode with a thinking budget (Qwen3 technical report, §4.3) on Qwen3-0.6B and GSM8K, as plain PyTorch.
# In thinking mode the model first writes its reasoning between <think> and </think>, then its final answer.
# A budget B caps the reasoning: if the model has not closed </think> after B tokens, the reasoning is halted and a
# stop-thinking instruction (ending in </think>) is inserted, and the model answers from what it has thought so far.
# Every sampled token is drawn from one seeded stream of uniforms per problem, so a token reproduces from its logits
# and the seed. Each "# %% [id]" section is one card of the tab.
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    temperature: float = 0.6  # Qwen3's card, thinking mode
    top_k: int = 20
    top_p: float = 0.95
    off_temperature: float = 0.7  # thinking off (enable_thinking=False)
    off_top_p: float = 0.8
    max_answer: int = 1024
    think_open: tuple = (151667, 198)  # "<think>", "\\n": the assistant turn starts thinking
    think_end: int = 151668  # "</think>"
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"
    # "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\\n</think>.\\n\\n"
    stop_thinking: tuple = (82796, 279, 7199, 882, 553, 279, 1196, 11, 358, 614, 311, 2968, 279, 6291, 3118, 389, 279,
                            7274, 5961, 1431, 624, 151668, 382)


# %% [sampling] Drawing one token
def sampling_probs(logits, temperature, top_k, top_p):
    """logits (..., V) → the distribution the next token is drawn from: divide by the temperature, keep the top_k
    largest, then drop the least likely tokens whose total probability is at most 1 − top_p."""
    scores = logits / temperature
    kth = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
    scores = scores.masked_fill(scores < kth, float("-inf"))
    ascending, order = scores.sort(dim=-1)
    remove = ascending.softmax(-1).cumsum(-1) <= 1 - top_p
    remove[..., -1:] = False  # always keep the most likely token
    scores = scores.masked_fill(remove.scatter(-1, order, remove), float("-inf"))
    return scores.softmax(-1)


def draw(probs, u):
    """Inverse CDF: lay the tokens out on [0, 1) from the most likely down (ties in id order); the token whose slice
    contains u is drawn. probs (..., V), u (...) → token ids (...)."""
    ordered, order = probs.sort(dim=-1, descending=True, stable=True)
    j = (ordered.cumsum(-1) <= u[..., None]).sum(-1)
    j = torch.minimum(j, (ordered > 0).sum(-1) - 1)
    return order.gather(-1, j[..., None])[..., 0]


def uniforms(seed, length):
    """One problem's stream: the t-th sampled token of a run, thinking or answer, uses u[t]."""
    return torch.rand(length, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# %% [budget] Budget forcing
def budget_context(prompt_ids, thinking_ids, budget, c: Config = Config()):
    """The context the final answer is written from, given the reasoning the model sampled with no budget.
    thinking_ids: the reasoning tokens, ending with </think> if the model closed it itself. Returns (context, the
    number of reasoning tokens kept, whether the budget cut the reasoning). If the model closed </think> before the
    budget, the budget changes nothing; otherwise its first \`budget\` tokens are kept and the instruction appended."""
    thinking = list(thinking_ids)
    ended = len(thinking) > 0 and thinking[-1] == c.think_end
    if ended and len(thinking) - 1 < budget:
        return list(prompt_ids) + list(c.think_open) + thinking, len(thinking) - 1, False
    kept = thinking[:budget]
    return list(prompt_ids) + list(c.think_open) + kept + list(c.stop_thinking), len(kept), True


# %% [answer] The final answer
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


def is_correct(text, gold):
    value = as_number(last_boxed(text))
    return value is not None and abs(value - float(gold)) < 1e-6


# %% [accuracy] Accuracy against tokens spent
def accuracy_and_tokens(runs):
    """runs: one setting's (correct, reasoning tokens, answer tokens) per problem → accuracy, mean tokens spent.
    Tokens spent counts what the model generated (reasoning + answer), not the inserted instruction."""
    correct = torch.tensor([float(r[0]) for r in runs])
    spent = torch.tensor([float(r[1] + r[2]) for r in runs])
    return correct.mean(), spent.mean()


# %% [model] Think, then answer
@torch.no_grad()
def sample_run(model, context, u, start, max_new, stops, temperature, top_k, top_p):
    """Sample from the context until a stop token: token t uses u[start + t]. No cache, for clarity."""
    ids = list(context)
    out = []
    for t in range(max_new):
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1].double()
        token = int(draw(sampling_probs(logits, temperature, top_k, top_p), u[start + t]))
        ids.append(token)
        out.append(token)
        if token in stops:
            break
    return out


def think_then_answer(model, prompt_ids, budget, seed, c: Config = Config(), max_thinking=2048):
    """Reason with at most \`budget\` tokens, then answer. Returns (reasoning tokens, answer tokens, cut)."""
    u = uniforms(seed, max_thinking + c.max_answer + 8)
    thinking = sample_run(model, list(prompt_ids) + list(c.think_open), u, 0, budget,
                          (c.think_end, *c.stops), c.temperature, c.top_k, c.top_p)
    context, kept, cut = budget_context(prompt_ids, thinking, budget, c)
    answer = sample_run(model, context, u, kept + (0 if cut else 1), c.max_answer, c.stops,
                        c.temperature, c.top_k, c.top_p)
    return thinking[:kept], answer, cut


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
# prompt = tokenizer.apply_chat_template([{"role": "user", "content": question + "\\n" + instruction}],
#                                        add_generation_prompt=True, enable_thinking=True)
# thinking, answer, cut = think_then_answer(model, prompt, budget=512, seed=gsm8k_row)
`;export{n as default};
