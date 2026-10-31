const e=`# Best-of-N with an outcome reward model (Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems") on GSM8K,
# as plain PyTorch: sample N answers from Qwen2.5-Math-1.5B-Instruct, score each whole answer with the reward model
# Skywork-Reward-V2-Qwen3-0.6B, and keep the highest-scoring one. Compared with a majority vote over the answers' final
# numbers (self-consistency) and with an oracle that knows the reference answer. The decoders are transformers' own
# (Qwen2ForCausalLM, Qwen3Model); this file is what the method adds around them.
# Each "# %% [id]" section is one card of the tab.
import itertools
import math
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    n: int = 16  # answers per problem
    temperature: float = 0.7
    top_k: int = 40  # the Self-consistency tab's samples, which this tab scores
    max_new_tokens: int = 512
    hidden_size: int = 1024  # the reward model's width (Qwen3-0.6B)
    system: str = "Please reason step by step, and put your final answer within \\\\boxed{}."


# %% [answer] Reading the final answer
def last_boxed(text):
    """Content of the last \\\\boxed{…} (braces may nest), or None when there is none or it is cut off."""
    start = text.rfind("\\\\boxed{")
    if start < 0:
        return None
    depth = 0
    for j in range(start + len("\\\\boxed"), len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[start + len("\\\\boxed{") : j]
    return None


def as_number(answer):
    """The boxed text as a number: units in \\\\text{…}, $, %, commas and spaces dropped; a/b, \\\\frac{a}{b} divided."""
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


def is_right(text, gold):
    value = as_number(last_boxed(text))
    return value is not None and abs(value - float(gold)) < 1e-6


# %% [reward-model] Outcome reward model
class OutcomeRewardModel(torch.nn.Module):
    """A decoder read at the conversation's last token, through one linear layer to a single number. It sees the
    question and one whole answer; the number is only meaningful relative to other answers' numbers."""

    def __init__(self, backbone, c: Config = Config()):
        super().__init__()
        self.backbone = backbone  # transformers' Qwen3Model: ids → last hidden states (B, T, H)
        self.score = torch.nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(-1) - 1  # right-padded: each conversation's final token
        return self.score(h[torch.arange(len(h)), last])[:, 0]  # (B,)


def conversation_ids(tokenizer, question, answer):
    """The reward model's input: its chat template over [user: question, assistant: answer], a duplicate BOS dropped."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}], tokenize=False
    )
    if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token) :]
    return tokenizer(text).input_ids


def right_pad(rows, pad_id):
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
    return ids, mask


# %% [best-of-n] Best-of-N: keep the highest score
def best_of_n(scores):
    """scores: (n,) → the index of the answer the reward model ranks first."""
    return int(torch.argmax(scores))


# %% [majority] Majority vote
def majority_vote(values):
    """values: the n answers' final numbers (None when unreadable, which does not vote). Returns the answers with the
    most votes (several on a tie) and the counts."""
    counts = {}
    for v in values:
        if v is not None:
            key = round(v, 6)
            counts[key] = counts.get(key, 0) + 1
    top = max(counts.values(), default=0)
    return [k for k, c in counts.items() if c == top], counts


# %% [curves] Accuracy against N
def best_of_n_accuracy(scores, right, n):
    """Chance that best-of-n is right when n of the N answers are drawn without replacement: the answer ranked r-th
    lowest (from 0) is the best of exactly C(r, n − 1) of the C(N, n) subsets."""
    order = torch.argsort(scores)
    N = len(scores)
    return sum(float(right[i]) * math.comb(r, n - 1) for r, i in enumerate(order.tolist())) / math.comb(N, n)


def oracle_accuracy(right, n):
    """pass@n: chance that at least one of n drawn answers is right (Chen et al. 2021)."""
    N, c = len(right), int(sum(right))
    return 1 - math.comb(N - c, n) / math.comb(N, n)


def majority_accuracy(values, gold, n):
    """Average over every subset of n answers; a tie counts as the chance of drawing the right one."""
    total = 0.0
    for subset in itertools.combinations(range(len(values)), n):
        winners, _ = majority_vote([values[i] for i in subset])
        total += sum(abs(w - float(gold)) < 1e-6 for w in winners) / len(winners) if winners else 0.0
    return total / math.comb(len(values), n)


# %% [model] Sample N answers, score them, choose
@torch.no_grad()
def sample_answers(generator, tokenizer, question, c: Config = Config()):
    messages = [{"role": "system", "content": c.system}, {"role": "user", "content": question}]
    prompt = torch.tensor([tokenizer.apply_chat_template(messages, add_generation_prompt=True)])
    out = generator.generate(prompt.expand(c.n, -1), attention_mask=torch.ones_like(prompt).expand(c.n, -1),
                             do_sample=True, temperature=c.temperature, top_k=c.top_k, top_p=1.0,
                             max_new_tokens=c.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(out[:, prompt.shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def choose(reward_model, rm_tokenizer, question, answers):
    """Every answer scored, alone in its row; returns the scores and the index best-of-N keeps."""
    ids, mask = right_pad([conversation_ids(rm_tokenizer, question, a) for a in answers], rm_tokenizer.pad_token_id)
    scores = reward_model(ids, mask)
    return scores, best_of_n(scores)


# generator = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
# rm = AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B", num_labels=1)
# reward_model = OutcomeRewardModel(rm.model); reward_model.score = rm.score
# answers = sample_answers(generator, tokenizer, question)
# scores, best = choose(reward_model, rm_tokenizer, question, answers)
`;export{e as default};
