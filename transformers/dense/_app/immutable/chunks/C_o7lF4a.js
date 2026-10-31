const e=`# Step-level beam search guided by a process reward model (as in Snell et al. 2024, "Scaling LLM Test-Time Compute
# Optimally can be More Effective than Scaling Model Parameters"), as plain PyTorch around transformers' generate:
# Qwen2.5-Math-1.5B-Instruct writes one reasoning step at a time (a step ends at a blank line), k candidates per beam;
# Skywork-o1-Open-PRM-Qwen-2.5-1.5B scores each candidate prefix; the best B survive and the rest are pruned. Finished
# answers keep their beam slots, so the search ends once B answers are finished. The PRM is pytorch/reasoning_prm.py.
# Each "# %% [id]" section is one card of the tab.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    beams: int = 4  # B: candidates kept at each depth
    expand: int = 4  # k: next steps sampled from each kept candidate
    max_depth: int = 24
    step_tokens: int = 192  # a step longer than this is cut and continued at the next depth
    temperature: float = 0.7
    top_k: int = 40  # as the Best-of-N answers were sampled
    stop: str = "\\n\\n"  # a step ends at a blank line


# %% [expand] Sampling the next steps
def end_of_step(new, tokenizer, eos_ids, c: Config = Config()):
    """Where a sampled continuation stops: at the end-of-turn token, or at the first token after which the new text
    contains a blank line (both kept), or at the token cap. Returns (ids, 'eos' | 'stop' | 'length')."""
    for t, token in enumerate(new):
        if token in eos_ids:
            return new[: t + 1], "eos"
        if c.stop in tokenizer.decode(new[: t + 1]):
            return new[: t + 1], "stop"
    return new, "length"


@torch.no_grad()
def expand(generator, tokenizer, prompt, prefixes, k, eos_ids, c: Config = Config()):
    """k continuations of every prefix (ids after the prompt) in one left-padded batch. Returns (parent, ids, end) per
    candidate, parents in order."""
    rows = [prompt + p for p in prefixes for _ in range(k)]
    width = max(map(len, rows))
    pad = tokenizer.pad_token_id
    ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows])
    mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])
    out = generator.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=c.temperature,
                             top_k=c.top_k, top_p=1.0, max_new_tokens=c.step_tokens, stop_strings=[c.stop],
                             tokenizer=tokenizer, eos_token_id=list(eos_ids), pad_token_id=pad)
    return [(row // k, *end_of_step(new, tokenizer, eos_ids, c)) for row, new in enumerate(out[:, width:].tolist())]


# %% [score] Scoring a partial solution
def prefix_score(piece_scores):
    """A candidate's score: the PRM's value at its last written line, its probability that the solution so far is right
    (pytorch/reasoning_prm.py; the trailing blank line is dropped before scoring)."""
    return float(piece_scores[-1])


# %% [select] Keeping the best B
def select(scores, ended, finished_before, c: Config = Config()):
    """scores: (m,) of this depth's candidates; ended: whether each ended the answer. The best B − (answers finished
    earlier) are kept: a kept candidate that ended the answer is finished, the others are beams; the rest are pruned.
    Returns (beams, finished, rank of every candidate, best first)."""
    order = torch.argsort(scores, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    kept = order[: c.beams - finished_before].tolist()
    return [i for i in kept if not ended[i]], [i for i in kept if ended[i]], rank


# %% [budget] What the search costs
def generated_tokens(nodes):
    """Every sampled step costs its new tokens, kept or pruned."""
    return sum(len(n["new"]) for n in nodes)


def prm_tokens(nodes):
    """The PRM reads every candidate's whole prefix once (without a cache shared between candidates)."""
    return sum(n["prmTokens"] for n in nodes)


def matched_n(search_tokens, mean_answer_tokens):
    """How many whole answers best-of-N could sample for the search's generated tokens."""
    return search_tokens / mean_answer_tokens


# %% [model] The search
@torch.no_grad()
def beam_search(sample_steps, score, c: Config = Config()):
    """sample_steps(prefixes, k) → [(parent, new ids, end)] (expand above, with the generator and prompt bound);
    score(ids) → (piece scores, PRM tokens) of the prompt's answer so far. Returns the answer node and every node."""
    nodes, beams, finished = [], [None], []  # None: the question itself
    for depth in range(1, c.max_depth + 1):
        prefixes = [nodes[b]["ids"] if b is not None else [] for b in beams]
        k = c.expand * (c.beams if depth == 1 else 1)  # the first depth samples B·k steps from the question
        start = len(nodes)
        for parent, new, end in sample_steps(prefixes, k):
            b = beams[parent]
            ids = (nodes[b]["ids"] if b is not None else []) + new
            pieces, read = score(ids)
            nodes.append({"parent": b, "depth": depth, "new": new, "ids": ids, "end": end,
                          "score": prefix_score(pieces), "prmTokens": read})
        fresh = list(range(start, len(nodes)))
        kept, done, rank = select(torch.tensor([nodes[i]["score"] for i in fresh]),
                                  [nodes[i]["end"] == "eos" for i in fresh], len(finished), c)
        for j, i in enumerate(fresh):
            nodes[i]["rank"] = int(rank[j])
            nodes[i]["state"] = "finished" if j in done else "kept" if j in kept else "pruned"
        finished += [fresh[j] for j in done]
        beams = [fresh[j] for j in kept]
        if not beams:
            break
    answer = max(finished or beams, key=lambda i: nodes[i]["score"])
    return answer, nodes


# generator, prm = Qwen2.5-Math-1.5B-Instruct, reasoning_prm.load(path); prompt = the chat prompt's ids
# sample_steps = lambda prefixes, k: expand(generator, tokenizer, prompt, prefixes, k, eos_ids=[151645, 151643])
# score = lambda ids: prm.step_scores(prm_tokenizer, question, tokenizer.decode(ids, skip_special_tokens=True))[0], ...
# answer, nodes = beam_search(sample_steps, score)
`;export{e as default};
