const e=`# Self-refine (Madaan et al. 2023) with Qwen2.5-Math-1.5B-Instruct on GSM8K, as plain PyTorch.
# One model answers, reviews its own answer, and revises it using that review, round after round: no second model,
# no reference answer, no training. The conversation is kept as the model's own token ids; each round appends a user
# turn asking for feedback, the model's feedback (ending in a verdict), a user turn asking for a revision, and the
# revised answer. Every turn is greedy. Two feedback requests are recorded: one asks the model to check its solution,
# the other (Self-Refine's framing) says there is an error. Each "# %% [id]" section is one card of the tab.
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    rounds: int = 3
    max_new: int = 512  # tokens per turn
    context: int = 4096  # the model's max_position_embeddings
    im_end: int = 151645  # "<|im_end|>"
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"


FEEDBACK = {
    "check": ('There may be a mistake in your solution above. Go through it step by step and check each step against '
              'the question. Explain any error you find. End your feedback with "The solution is correct." or "The '
              'solution is incorrect."'),
    "error": ('There is an error in your solution above because of a lack of understanding of the question. What is '
              'the error? To find it, go through the solution step by step and check whether everything looks good. '
              'End your feedback with "The solution is correct." if everything looks good, or "The solution is '
              'incorrect." if it is not.'),
}
REVISE = ('Based on your feedback, solve the problem again. Please reason step by step, and put your final answer '
          'within \\\\boxed{}.')
# a user turn as Qwen2.5's chat template writes it after an assistant turn:
# turn_ids = tokenizer("\\n<|im_start|>user\\n" + text + "<|im_end|>\\n<|im_start|>assistant\\n").input_ids


# %% [greedy] Taking the most likely token
def greedy(logits):
    """logits (..., V) → the most likely token (ties to the lower id) and its probability under the model."""
    token = logits.argmax(-1)  # the first maximum: the lowest id
    return token, logits.softmax(-1).gather(-1, token[..., None])[..., 0]


# %% [feedback] Feedback on its own answer
def closed(turn, c: Config = Config()):
    """A turn as it stays in the conversation: one cut at its token limit is closed with <|im_end|>."""
    return list(turn) if turn and turn[-1] in c.stops else list(turn) + [c.im_end]


def next_context(history, turn_ids):
    """The context the model writes its next turn from: everything so far, then the user turn asking for it."""
    return list(history) + list(turn_ids)


# %% [verdict] Reading the verdict
VERDICT = re.compile(r"solution is (correct|incorrect|not correct|wrong)", re.IGNORECASE)


def verdict(feedback_text):
    """'correct', 'wrong' or None: the last "The solution is …" in the feedback, markdown stars ignored."""
    found = VERDICT.findall(feedback_text.replace("*", ""))
    if not found:
        return None
    return "correct" if found[-1].lower() == "correct" else "wrong"


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


# %% [rounds] Accuracy, fixes and breaks per round
def round_effects(correct):
    """correct: (rounds + 1, problems) booleans, round 0 the first answer → per round its accuracy, and from round 1
    on how many wrong answers the revision fixed and how many right ones it broke."""
    correct = torch.as_tensor(correct)
    accuracy = correct.double().mean(-1)
    fixed = (~correct[:-1] & correct[1:]).sum(-1)
    broken = (correct[:-1] & ~correct[1:]).sum(-1)
    return accuracy, fixed, broken


def stop_at_correct(verdicts, correct, tokens):
    """Self-refine's stopping rule: keep the latest answer once a feedback calls it correct. verdicts[r][q] is round
    r's feedback (r = 1 …; index 0 unused), correct[r][q] round r's answer, tokens[r][q] the tokens round r
    generated (feedback + revision; round 0: the answer). Returns accuracy and mean tokens spent."""
    rounds, n = len(correct) - 1, len(correct[0])
    right, spent = 0, 0
    for q in range(n):
        r = next((k for k in range(1, rounds + 1) if verdicts[k][q] == "correct"), None)
        keep = rounds if r is None else r - 1
        right += bool(correct[keep][q])
        spent += sum(tokens[k][q][0] + tokens[k][q][1] for k in range(keep + 1)) + (tokens[r][q][0] if r else 0)
    return right / n, spent / n


# %% [model] The self-refine loop
@torch.no_grad()
def generate(model, context, max_new, c: Config = Config()):
    """Greedy continuation until a stop token. No cache, for clarity."""
    ids, out = list(context), []
    for _ in range(max_new):
        token, _ = greedy(model(input_ids=torch.tensor([ids])).logits[0, -1])
        ids.append(int(token))
        out.append(int(token))
        if out[-1] in c.stops:
            break
    return out


def self_refine(model, prompt_ids, feedback_ids, revise_ids, c: Config = Config()):
    """Answer, then \`rounds\` times: feedback on the latest answer, a revised answer. Returns the turns in order."""
    history, turns = list(prompt_ids), []
    answer = generate(model, history, min(c.max_new, c.context - len(history)), c)
    history += closed(answer, c)
    turns.append(answer)
    for _ in range(c.rounds):
        for user in (feedback_ids, revise_ids):
            context = next_context(history, user)
            turn = generate(model, context, min(c.max_new, c.context - len(context)), c)
            history = context + closed(turn, c)
            turns.append(turn)
    return turns


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", dtype=torch.float32)
# prompt = tokenizer.apply_chat_template([{"role": "system", "content": instruction},
#                                         {"role": "user", "content": question}], add_generation_prompt=True)
# turns = self_refine(model, prompt, turn_ids(FEEDBACK["error"]), turn_ids(REVISE))
`;export{e as default};
