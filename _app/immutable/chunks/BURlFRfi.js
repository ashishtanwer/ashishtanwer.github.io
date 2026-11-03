const e=`# GRPO, DAPO and RLOO side by side, the Group RL Compare tab, as plain PyTorch.
# All three sample a group of answers per problem, score them with rule-based rewards and push up the answers that beat
# the group's baseline, with no value model. They differ in a few formulas: the baseline and its scale, whether the
# ratio π_θ / π_old is taken per token or per answer and where it is clipped, how the per-token losses are averaged,
# what is added to the reward, and which groups are trained on. The tab runs all three from Qwen2.5-0.5B-Instruct on
# GSM8K test rows 1 and 3 with the same seeds: 4 rounds of 8 answers per problem, 2 AdamW updates per round.
# Each "# %% [id]" section is one card of the tab.
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    group_size: int = 8  # answers per problem (G, RLOO's k)
    max_new_tokens: int = 320  # L_max
    eps_low: float = 0.2  # all three: a ratio below 1 − 0.2 stops counting where the advantage is negative
    eps_high: dict = None  # the upper clip: GRPO 0.2, DAPO 0.28 (clip-higher), RLOO 0.2 on the whole answer
    grpo_beta: float = 0.04  # GRPO's KL coefficient (DeepSeekMath's); DAPO and RLOO have no KL term
    overlong_cache: int = 64  # DAPO: the penalty starts L_cache tokens before L_max
    iterations: int = 2  # updates per sampled batch (μ), for all three here
    lr: float = 3e-6
    max_grad_norm: float = 1.0
    reward_weights: tuple = (1.0, 0.5)  # correct, boxed; DAPO adds the overlong term with weight 1

    def __post_init__(self):
        self.eps_high = self.eps_high or {"grpo": 0.2, "dapo": 0.28, "rloo": 0.2}


# %% [advantages] Three advantages from the same rewards
def last_boxed(text):
    found = re.findall(r"\\\\boxed\\{([^{}]*)\\}", text)
    return found[-1] if found else None


def rewards(text, length, gold, method, c: Config = Config()):
    """(correct, boxed, overlong) and the total this method trains on: 1 if the last \\\\boxed{} holds the reference
    number, 1 if there is a \\\\boxed{} at all, weighted 1 and 0.5; DAPO adds its overlong penalty."""
    found = last_boxed(text)
    try:
        value = float(found.replace(",", "").replace("$", "").strip()) if found is not None else None
    except ValueError:
        value = None
    correct = 1.0 if value is not None and value == float(gold) else 0.0
    boxed = 1.0 if found is not None else 0.0
    overlong = overlong_penalty(length, c)
    total = c.reward_weights[0] * correct + c.reward_weights[1] * boxed + (overlong if method == "dapo" else 0.0)
    return correct, boxed, overlong, total


def advantages(total, method):
    """total: (G,) one group's rewards. GRPO and DAPO: Â_i = (r_i − mean) / (std + 1e-4), so every group's advantages
    have the same spread. RLOO: A_i = r_i − (Σ_j r_j − r_i) / (G − 1), the mean of the other answers, which does not
    depend on answer i; it equals G / (G − 1) · (r_i − mean): the same signs as GRPO's, in reward units."""
    if method == "rloo":
        return total - (total.sum() - total) / (len(total) - 1)
    return (total - total.mean()) / (total.std() + 1e-4)


# %% [overlong] DAPO's overlong penalty
def overlong_penalty(length, c: Config = Config()):
    """0 up to L_max − L_cache tokens, then a linear ramp to −1 at L_max (DAPO's Eq. 13): long answers lose reward
    gradually instead of only by being cut off. GRPO's and RLOO's rewards ignore length."""
    start = c.max_new_tokens - c.overlong_cache
    if length <= start:
        return 0.0
    if length <= c.max_new_tokens:
        return (start - length) / c.overlong_cache
    return -1.0


# %% [dynamic] Dynamic sampling
def keep_group(correct, method):
    """correct: (G,) 0/1. DAPO drops a group whose answers are all right or all wrong (its correctness carries no
    signal) and samples the next problem in its place; GRPO and RLOO train on every group they sampled, where such a
    group has advantage 0 (or only the boxed reward's spread)."""
    right = int(correct.sum())
    return method != "dapo" or 0 < right < correct.numel()


# %% [ratio] One ratio per token, or one per answer
def ratios(logps, old_logps, mask, method):
    """logps, old_logps: (n, L) per-token log π_θ and log π_old; mask: (n, L). GRPO and DAPO: ρ_t = π_θ(o_t) / π_old(o_t)
    for every token. RLOO: the whole answer is one action, ρ = π_θ(o) / π_old(o) = exp(Σ_t (log π_θ − log π_old)), the
    product of all the per-token ratios, so it leaves 1 much faster than any one of them."""
    if method == "rloo":
        return torch.exp(((logps - old_logps) * mask).sum(-1))
    return torch.exp(logps - old_logps) * mask


def clipped(ratio, advantages, method, c: Config = Config()):
    """Where the clipped surrogate takes the clipped branch (the gradient is 0 there): ρ < 1 − ε_low with A < 0, or
    ρ > 1 + ε_high with A > 0. Per token for GRPO and DAPO, per answer for RLOO."""
    a = advantages if method == "rloo" else advantages[:, None]
    return ((ratio < 1 - c.eps_low) & (a < 0)) | ((ratio > 1 + c.eps_high[method]) & (a > 0))


# %% [average] Three ways to average the loss
def loss(method, logps, old_logps, advantages, mask, tokens_in_batch, answers_in_batch, ref_logps=None,
         c: Config = Config()):
    """One micro-batch's share of the batch loss (the shares add up to the batch's). Per answer or token, the clipped
    surrogate −min(ρ A, clip(ρ, 1 − ε_low, 1 + ε_high) A). GRPO adds β · KL (k3 estimate) per token, averages each
    answer's tokens, then the answers: a token of a short answer weighs more. DAPO divides the sum over all tokens by
    the batch's token count: every token weighs the same. RLOO's ρ multiplies Σ_t log π, so every token of an answer
    carries its answer's whole A; the answers are averaged."""
    eps = (1 - c.eps_low, 1 + c.eps_high[method])
    if method == "rloo":
        rho = ratios(logps, old_logps, mask, method)
        return (-torch.min(rho * advantages, torch.clamp(rho, *eps) * advantages)).sum() / answers_in_batch
    rho = torch.exp(logps - old_logps)
    a = advantages[:, None]
    per_token = -torch.min(rho * a, torch.clamp(rho, *eps) * a)
    if method == "grpo":
        per_token = per_token + c.grpo_beta * (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
        return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).sum() / answers_in_batch
    return (per_token * mask).sum() / tokens_in_batch


# %% [entropy] Entropy of the next-token distribution
def answer_logps(model, prompt_ids, completion_ids, completion_mask):
    """log π(o_t | q, o_<t) at every answer position, and the entropy H_t = −Σ_v π(v) log π(v) of the whole next-token
    distribution there (what TRL logs as "entropy"). prompt_ids: (1, n), shared by the group; completion_ids and
    completion_mask: (G, L), right-padded after each answer's end. Sampling at temperature 1 draws o_t from exactly this
    distribution; DAPO's clip-higher is there to keep H from collapsing."""
    g, length = completion_ids.shape
    ids = torch.cat([prompt_ids.expand(g, -1), completion_ids], dim=1)
    mask = torch.cat([torch.ones(g, prompt_ids.shape[1], dtype=torch.long), completion_mask], dim=1)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, -length - 1 : -1]  # the positions that predict o_t
    log_probs = logits.log_softmax(dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return log_probs.gather(-1, completion_ids[..., None])[..., 0], entropy


# %% [model] One update for any of the three
def update(method, policy, optimizer, batches, reference=None, c: Config = Config()):
    """batches: micro-batches of (prompt_ids, completion_ids, completion_mask, advantages, old_logps); their losses add
    up to the batch loss, so the gradients accumulate. GRPO also needs the frozen reference policy."""
    tokens = sum(int(mask.sum()) for _, _, mask, _, _ in batches)
    answers = sum(len(mask) for _, _, mask, _, _ in batches)
    optimizer.zero_grad()
    total = 0.0
    for prompt_ids, completion_ids, completion_mask, adv, old_logps in batches:
        logps, _ = answer_logps(policy, prompt_ids, completion_ids, completion_mask)
        ref = None
        if method == "grpo":
            with torch.no_grad():
                ref, _ = answer_logps(reference, prompt_ids, completion_ids, completion_mask)
        part = loss(method, logps, old_logps, adv, completion_mask.float(), tokens, answers, ref, c)
        part.backward()
        total += part.detach()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
    optimizer.step()
    return total


# for round in range(4):  # the same seed per round for all three: round 1 is the same answers
#     groups = [sample(policy, problem, G) for problem in problems]  # DAPO: drop and refill (keep_group)
#     batches = [(prompt, ids, mask, advantages(total, method), old_logps) for ...]
#     for _ in range(c.iterations):
#         update(method, policy, optimizer, batches, reference)
`;export{e as default};
