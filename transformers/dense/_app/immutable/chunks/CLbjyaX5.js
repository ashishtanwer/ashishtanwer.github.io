const t=`# MMLU (Hendrycks et al. 2021) scored by log-likelihood, on Qwen2.5-0.5B-Instruct, as plain PyTorch.
# A question has four options and one right answer. The model never generates: each option is scored by the
# probability the model gives it, and the most probable option is the model's answer. Three prompt formats: the
# options lettered A–D and the next token scored (" A" … " D"), the same in the chat template ("A" … "D" as the
# assistant's first token), or no options shown and each option's text scored as a continuation (cloze). A k-shot
# prompt puts k answered questions of the same subject first. Each "# %% [id]" section is one card of the tab.
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    shots: int = 5  # answered dev questions before the test question (MMLU's dev split has five per subject)
    letters: str = "ABCD"
    header: str = "The following are multiple choice questions (with answers) about {subject}.\\n\\n"
    instruction: str = "Answer with the letter of the correct option."
    system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."  # Qwen2.5's default


# %% [prompt] Building the prompt
def lettered(q, c: Config = Config()):
    """A question and its options, lettered."""
    return q["question"].strip() + "".join(f"\\n{c.letters[i]}. {o}" for i, o in enumerate(q["choices"]))


def letter_prompt(subject, shots, q, c: Config = Config()):
    """The original MMLU prompt: header, the answered shots, then the test question ending at "Answer:"."""
    text = c.header.format(subject=subject.replace("_", " "))
    for s in shots:
        text += lettered(s, c) + f"\\nAnswer: {c.letters[s['answer']]}\\n\\n"
    return text + lettered(q, c) + "\\nAnswer:"


def chat_prompt(subject, shots, q, c: Config = Config()):
    """The same questions as chat turns in Qwen's ChatML template, ready for the assistant's first token."""
    turn = lambda role, text: f"<|im_start|>{role}\\n{text}<|im_end|>\\n"
    text = turn("system", c.system)
    for i, item in enumerate([*shots, q]):
        head = c.header.format(subject=subject.replace("_", " ")) if i == 0 else ""
        text += turn("user", head + lettered(item, c) + "\\n" + c.instruction)
        if item is not q:
            text += turn("assistant", c.letters[item["answer"]])
    return text + "<|im_start|>assistant\\n"


def cloze_prompt(subject, shots, q, c: Config = Config()):
    """No options: each shot answered by its option's text; each option of the test question is a continuation."""
    text = c.header.format(subject=subject.replace("_", " "))
    for s in shots:
        text += f"Question: {s['question'].strip()}\\nAnswer: {s['choices'][s['answer']]}\\n\\n"
    return text + f"Question: {q['question'].strip()}\\nAnswer:"


# %% [score] Scoring the options
def letter_choice(letter_logp):
    """(..., 4) log-probabilities of the four letters → the chosen option (ties to the earlier letter)."""
    return letter_logp.argmax(-1)


def cloze_choice(total_logp, chars=None):
    """(..., 4) total log-probability of each option's continuation → the chosen option; with \`chars\` (the
    continuation's length in characters) the total is divided by it first (lm-evaluation-harness's acc_norm)."""
    return (total_logp if chars is None else total_logp / chars).argmax(-1)


# %% [accuracy] Accuracy and its standard error
def accuracy(choices, answers):
    """Share of right answers and its standard error over the questions, sqrt(p (1 − p) / (n − 1))."""
    right = (choices == answers).double()
    p, n = right.mean().item(), right.numel()
    return p, math.sqrt(p * (1 - p) / (n - 1))


# %% [model] Scoring with the model
@torch.no_grad()
def next_token_logp(model, prompt_ids):
    """log p of every vocabulary token as the next token after the prompt."""
    return model(input_ids=torch.tensor([prompt_ids])).logits[0, -1].float().log_softmax(-1)


@torch.no_grad()
def continuation_logp(model, context_ids, continuation_ids):
    """log p of each continuation token given the context and the continuation tokens before it."""
    ids = torch.tensor([context_ids + continuation_ids])
    logp = model(input_ids=ids).logits[0, len(context_ids) - 1 : -1].float().log_softmax(-1)
    return logp.gather(-1, torch.tensor(continuation_ids)[:, None])[:, 0]


def answer(model, tokenizer, subject, shots, q, form="letter", c: Config = Config()):
    """The option the model picks for question q in one format, with the scores behind it."""
    encode = lambda text: tokenizer(text, add_special_tokens=False).input_ids
    if form in ("letter", "chat"):
        prompt = (letter_prompt if form == "letter" else chat_prompt)(subject, shots, q, c)
        letter_ids = [encode((" " if form == "letter" else "") + x)[0] for x in c.letters]
        scores = next_token_logp(model, encode(prompt))[letter_ids]
        return int(letter_choice(scores)), scores
    context = cloze_prompt(subject, shots, q, c)
    totals = []
    for option in q["choices"]:
        whole, ctx = encode(context + " " + option), encode(context)
        totals.append(continuation_logp(model, ctx, whole[len(ctx) :]).sum())
    totals = torch.stack(totals)
    return int(cloze_choice(totals)), totals


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
# dev = mmlu_dev_rows(subject)  # the subject's five answered dev questions
# choice, scores = answer(model, tokenizer, subject, dev[:5], test_row, form="letter")
`;export{t as default};
