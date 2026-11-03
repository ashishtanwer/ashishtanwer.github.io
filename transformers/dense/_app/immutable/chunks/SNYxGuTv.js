const e=`# Repeated sampling compared: the majority vote (self-consistency, Wang et al. 2023) and the reward model's pick
# (best-of-N, Cobbe et al. 2021) on the same sampled answers, as plain PyTorch. Both rules choose among N answers that
# Qwen2.5-Math-1.5B-Instruct sampled for one GSM8K problem: the vote reads only their final numbers, best-of-N runs
# the outcome reward model Skywork-Reward-V2-Qwen3-0.6B once per answer and keeps the highest score. Sampling and
# scoring are the Self-consistency and Best-of-N tabs' code; this file is how the page sets them side by side.
# Each "# %% [id]" section is one card of the tab.
import itertools
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    shared: int = 16  # answers both rules choose among (the reward model scored the first 16)
    samples: int = 32  # answers the vote can use (the Self-consistency tab's)
    generator_parameters: int = 1_543_714_304  # Qwen2.5-Math-1.5B-Instruct
    reward_parameters: int = 596_050_944  # Skywork-Reward-V2-Qwen3-0.6B


# %% [vote] Majority vote
def vote(keys, gold):
    """keys: each answer's final number as a string (None: no number, so no vote). Returns the winners (several on a
    tie) and the chance the vote is right: 1 or 0, or on a tie the chance of drawing the right winner."""
    counts = {}
    for key in keys:
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return [], 0.0
    top = max(counts.values())
    winners = [key for key, count in counts.items() if count == top]
    return winners, winners.count(gold) / len(winners)


# %% [reward-pick] Best-of-N: the highest reward
def reward_pick(scores):
    """scores: (n,) the reward model's score of each answer → the index it ranks first. The scores of different
    problems are not comparable; only their order within one problem matters."""
    return int(torch.argmax(scores))


# %% [curves] Accuracy against N
def accuracy_at(keys, scores, gold, n):
    """One problem's chance that each rule is right when n of its answers are drawn without replacement, averaged
    over all C(N, n) subsets: (vote, reward pick, any answer right). The last is the ceiling for any rule."""
    right = [key is not None and key == gold for key in keys]
    voted = picked = ceiling = 0.0
    for subset in itertools.combinations(range(len(keys)), n):
        voted += vote([keys[i] for i in subset], gold)[1]
        picked += right[subset[reward_pick(scores[list(subset)])]]
        ceiling += any(right[i] for i in subset)
    total = math.comb(len(keys), n)
    return torch.tensor([voted, picked, ceiling], dtype=torch.float64) / total


# %% [agreement] Where the two rules agree
def agreement(keys, scores, gold):
    """One problem, all n answers: what each rule chooses, which is right, and whether they choose the same answer."""
    winners, vote_right = vote(keys, gold)
    pick = reward_pick(scores)
    return {
        "winners": winners,
        "vote_right": vote_right,
        "pick": pick,
        "pick_right": keys[pick] is not None and keys[pick] == gold,
        "same": len(winners) == 1 and keys[pick] == winners[0],
    }


# %% [compute] What each rule spends
def compute(prompt_tokens, written_tokens, reward_tokens, c: Config = Config()):
    """One problem's arithmetic, ≈ 2 × parameters × tokens through each model. Per answer, the generator reads the
    prompt and writes the answer; best-of-N adds one reward-model pass over the question and the answer. Lists hold
    one entry per answer; returns (vote, best-of-N) in FLOPs."""
    generator = 2 * c.generator_parameters * (sum(prompt_tokens) + sum(written_tokens))
    return generator, generator + 2 * c.reward_parameters * sum(reward_tokens)


# %% [model] Both rules on one problem
def compare(keys, scores, gold, c: Config = Config()):
    """The vote and the pick over the first \`shared\` answers (the ones with scores), and the vote over all of them."""
    shared = agreement(keys[: c.shared], scores[: c.shared], gold)
    _, vote_all = vote(keys[: c.samples], gold)
    return shared, vote_all


# keys = [final number of each of the 32 answers]; scores = reward model scores of the first 16 (Best-of-N tab)
# shared, vote_all = compare(keys, scores, gold)
# curve = torch.stack([accuracy_at(keys[:16], scores, gold, n) for n in range(1, 17)])  # per problem; average them
`;export{e as default};
