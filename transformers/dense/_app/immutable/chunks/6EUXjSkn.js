const e=`# An LLM as a pairwise judge (Zheng et al. 2023, MT-Bench), with Qwen3-4B-Instruct-2507, as plain PyTorch.
# The judge reads a question and two answers labelled A and B, explains which is better and ends with "[[A]]",
# "[[B]]" or "[[C]]" (a tie). Judges prefer one position, so every pair is judged twice with the answers swapped;
# a pair's verdict is a win only if both orders agree. Agreement with human experts is counted vote by vote.
# Each "# %% [id]" section is one card of the tab.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    max_new_tokens: int = 1024
    stops: tuple = (151645, 151643)  # "<|im_end|>", "<|endoftext|>"
    letters: tuple = (32, 33, 34)  # "A", "B", "C" in Qwen3's vocabulary
    system: str = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to "
        "the user question displayed below. You should choose the assistant that follows the user's instructions and "
        "answers the user's question better. Your evaluation should consider factors such as the helpfulness, "
        "relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
        "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the "
        "order in which the responses were presented does not influence your decision. Do not allow the length of "
        "the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective "
        "as possible. After providing your explanation, output your final verdict by strictly following this "
        "format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better, and \\"[[C]]\\" for a tie.")
    template: str = ("[User Question]\\n{question}\\n\\n[The Start of Assistant A's Answer]\\n{answer_a}\\n[The End of "
                     "Assistant A's Answer]\\n\\n[The Start of Assistant B's Answer]\\n{answer_b}\\n[The End of Assistant "
                     "B's Answer]")


# %% [prompt] The judge's prompt, in both orders
def judge_messages(question, first_answer, second_answer, c: Config = Config()):
    """MT-Bench's pair-v2 prompt as chat messages: the system prompt and the question with the two answers."""
    user = c.template.format(question=question, answer_a=first_answer, answer_b=second_answer)
    return [{"role": "system", "content": c.system}, {"role": "user", "content": user}]


def both_orders(question, answers, pair):
    """The pair judged twice: pair[0]'s answer as A, then as B."""
    return [judge_messages(question, answers[pair[0]], answers[pair[1]]),
            judge_messages(question, answers[pair[1]], answers[pair[0]])]


# %% [verdict] Reading the verdict
def verdict(judgment):
    """FastChat's reading: [[A]] if the judgment contains it, else [[B]], else [[C]] (a tie), else an error."""
    for mark, result in (("[[A]]", "A"), ("[[B]]", "B"), ("[[C]]", "tie")):
        if mark in judgment:
            return result
    return "error"


def winner(result, first, second):
    """A judgment's verdict as the model it picks, given which model's answer was A."""
    return {"A": first, "B": second}.get(result, result)


def combined(order1, order2):
    """The pair's verdict: a win only if both orders name the same winner, else a tie (an error drops the pair)."""
    if "error" in (order1, order2):
        return "error"
    return order1 if order1 == order2 and order1 != "tie" else "tie"


# %% [agreement] Position bias and agreement with humans
def position_bias(orders, pair):
    """orders: per pair, the winners of (pair[0] as A, pair[1] as A) → how often both orders agree, and how often
    each order picks whichever answer it saw first (or second): the flips position bias causes."""
    valid = [(a, b) for a, b in orders if "error" not in (a, b)]
    first = sum(a == pair[0] and b == pair[1] for a, b in valid)
    second = sum(a == pair[1] and b == pair[0] for a, b in valid)
    return {"consistent": sum(a == b for a, b in valid), "biasedFirst": first, "biasedSecond": second,
            "pairs": len(valid)}


def agreement(judge_votes, human_votes, without_ties=False):
    """Every human vote on a pair against the judge's verdict on it: (agreeing votes, votes counted). Without ties,
    pairs the judge calls a tie and tie votes are left out."""
    agree = total = 0
    for key, vote in judge_votes.items():
        if key not in human_votes or vote == "error" or (without_ties and vote == "tie"):
            continue
        for v in human_votes[key]:
            if without_ties and v == "tie":
                continue
            total += 1
            agree += v == vote
    return agree, total


# %% [model] Judging greedily
@torch.no_grad()
def judge(model, prompt_ids, c: Config = Config()):
    """Greedy decoding (no cache) to a stop token; per step the generated token and the full-vocabulary log-softmax
    row of the three verdict letters, so the verdict's probabilities can be read where "[[" was written."""
    ids, out, letters = list(prompt_ids), [], []
    for _ in range(c.max_new_tokens):
        logp = model(input_ids=torch.tensor([ids])).logits[0, -1].float().log_softmax(-1)
        token = int(logp.argmax())
        letters.append(logp[list(c.letters)])
        ids.append(token)
        out.append(token)
        if token in c.stops:
            break
    return out, torch.stack(letters)


# judge_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16)
# for messages in both_orders(question, answers, ("alpaca-13b", "vicuna-13b-v1.2")):
#     ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
#     tokens, letter_logp = judge(judge_model, ids)
#     print(verdict(tokenizer.decode(tokens, skip_special_tokens=True)))
`;export{e as default};
