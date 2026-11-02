const e=`# Budget forcing (s1, Muennighoff et al. 2025) on Qwen3-0.6B and GSM8K, as plain PyTorch.
# A reasoning model writes its thinking between <think> and </think>, then its answer. Budget forcing edits the
# thinking as it is written. To make the model think longer, each time it draws an end of its thinking (</think>, or
# the end of its turn) that token is replaced by "Wait" and the end is banned for one step, so it goes on reasoning.
# To make it stop, "\\n</think>\\n\\n" is forced after B reasoning tokens and the model answers from what it has.
# Every sampled token of a problem is drawn from one seeded stream of uniforms, so a token reproduces from its logits
# and the seed. Each "# %% [id]" section is one card of the tab.
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    temperature: float = 0.6  # Qwen3's card, thinking mode
    top_k: int = 20
    top_p: float = 0.95
    waits: int = 4  # at most this many "Wait"s per problem
    cap: int = 4096  # reasoning tokens, Waits included; the end is forced there
    max_answer: int = 1024
    think_open: tuple = (151667, 198)  # "<think>", "\\n": the assistant turn starts thinking
    think_end: int = 151668  # "</think>"
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"
    wait: tuple = (14190,)  # "Wait"
    forced_end: tuple = (198, 151668, 271)  # "\\n</think>\\n\\n", as Qwen3's chat template writes the delimiter


# %% [sampling] Drawing one token, and the ban after "Wait"
def ban_ends(logits, ends):
    """s1's min_tokens = 1 after "Wait": the ends of the thinking may not be the next token. logits (..., V)."""
    return logits.index_fill(-1, torch.tensor(ends), float("-inf"))


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
    """One problem's stream: the t-th token the model draws, in its reasoning or an answer, uses u[t]."""
    return torch.rand(length, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# %% [wait] "Wait" instead of the end
def chain_with_waits(segments, c: Config = Config()):
    """The reasoning after budget forcing, from what the model sampled in each stretch between two Waits. segments:
    lists of drawn tokens; every one but the last ends with the end the model drew (</think> or the end of its
    turn), which "Wait" replaces. Returns the reasoning tokens and the positions of the Waits in them."""
    tokens, waits = [], []
    for i, drawn in enumerate(segments):
        last = i + 1 == len(segments)
        ended = len(drawn) > 0 and drawn[-1] in (c.think_end, *c.stops)
        tokens += drawn[:-1] if ended else drawn
        if not last:
            assert ended, "a Wait replaces an end the model drew"
            waits.append(len(tokens))
            tokens += list(c.wait)
    return tokens, waits


def answer_context(prompt_ids, tokens, end_at, end_token, c: Config = Config()):
    """The context the answer after an end is written from: the reasoning up to the end, then the model's own
    </think>, or the forced delimiter where the reasoning reached the cap or the model ended its turn instead."""
    tail = [c.think_end] if end_token == c.think_end else list(c.forced_end)
    return list(prompt_ids) + list(c.think_open) + list(tokens[:end_at]) + tail


# %% [budget] Forcing the end at a budget
def cut_context(prompt_ids, tokens, first_end_at, first_end_token, budget, c: Config = Config()):
    """The context the answer under a budget is written from, given the reasoning without a budget and its first
    end. Returns (context, reasoning tokens kept, whether the budget cut). If the model wrote </think> before the
    budget, the budget changes nothing; otherwise the first \`budget\` tokens are kept and the delimiter forced."""
    if first_end_token == c.think_end and first_end_at is not None and first_end_at < budget:
        return answer_context(prompt_ids, tokens, first_end_at, c.think_end, c), first_end_at, False
    return list(prompt_ids) + list(c.think_open) + list(tokens[:budget]) + list(c.forced_end), budget, True


# %% [end-probability] How much the model wants to stop
@torch.no_grad()
def end_probability(model, context, start, c: Config = Config()):
    """P(</think>) as the next token after every prefix context[:i], i = start … len(context), from one forward pass:
    how close the model is to ending its reasoning at each step. Returns len(context) − start + 1 values."""
    logits = model(input_ids=torch.tensor([context])).logits[0, start - 1 :].double()
    return logits.softmax(-1)[:, c.think_end]


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
    """runs: one setting's (correct, reasoning tokens sampled, answer tokens) per problem → accuracy, mean tokens.
    Tokens spent count what the model generated: not the inserted "Wait"s or delimiter."""
    correct = torch.tensor([float(r[0]) for r in runs])
    spent = torch.tensor([float(r[1] + r[2]) for r in runs])
    return correct.mean(), spent.mean()


def fixed_and_broken(before, after):
    """Correctness per problem before and after one more Wait → (wrong turned right, right turned wrong)."""
    before, after = torch.tensor(before), torch.tensor(after)
    return int((~before & after).sum()), int((before & ~after).sum())


# %% [model] Think with Waits, then answer
@torch.no_grad()
def sample(model, context, u, start, max_new, stops, banned_steps, c: Config = Config()):
    """Sample from the context until a stop token: token t uses u[start + t], and the ends are banned for the first
    banned_steps tokens. No cache, for clarity."""
    ids, out = list(context), []
    for t in range(max_new):
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1].double()
        if t < banned_steps:
            logits = ban_ends(logits, (c.think_end, *c.stops))
        token = int(draw(sampling_probs(logits, c.temperature, c.top_k, c.top_p), u[start + t]))
        ids.append(token)
        out.append(token)
        if token in stops:
            break
    return out


def budget_force(model, prompt_ids, waits, seed, c: Config = Config()):
    """Reason, replacing the first \`waits\` ends with "Wait" (until the cap), then answer. Returns (reasoning tokens
    with the Waits, their positions, the answer tokens)."""
    u = uniforms(seed, c.cap + c.waits + 1 + c.max_answer + 8)
    ends = (c.think_end, *c.stops)
    segments, drawn, banned = [], 0, 0  # drawn: uniforms used so far
    while True:
        tokens, _ = chain_with_waits(segments + [[]], c)  # what the model has so far (a Wait after each segment)
        room = c.cap - len(tokens)
        new = sample(model, list(prompt_ids) + list(c.think_open) + tokens, u, drawn, room, ends, banned, c)
        segments.append(new)
        drawn += len(new)
        ended = len(new) > 0 and new[-1] in ends
        if not ended or len(segments) > waits or len(tokens) + len(new) - 1 + len(c.wait) >= c.cap:
            break
        banned = 1
    tokens, positions = chain_with_waits(segments, c)
    end_token = segments[-1][-1] if segments[-1] and segments[-1][-1] in ends else None
    context = answer_context(prompt_ids, tokens, len(tokens), end_token, c)
    answer = sample(model, context, u, drawn, c.max_answer, c.stops, 0, c)
    return tokens, positions, answer


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32)
# prompt = tokenizer.apply_chat_template([{"role": "user", "content": question + "\\n" + instruction}],
#                                        add_generation_prompt=True, enable_thinking=True)
# reasoning, waits, answer = budget_force(model, prompt, waits=2, seed=gsm8k_row)
`;export{e as default};
