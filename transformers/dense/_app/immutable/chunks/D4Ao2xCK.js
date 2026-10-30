const e=`# DAPO (Yu et al. 2025) on Qwen2.5-0.5B-Instruct and GSM8K, the DAPO tab, as plain PyTorch.
# DAPO is GRPO with four changes for long reasoning answers: clip the ratio higher than lower (ε_high > ε_low) so
# unlikely tokens of good answers can still grow; drop groups whose answers all got the same correctness (their
# advantages are 0) and sample more; average the loss over all tokens of the batch rather than per answer; and shape
# the reward of overlong answers instead of truncating silently. It also drops the KL term. The tab's run keeps the GRPO
# tab's model, problems, rewards and sampling; each "# %% [id]" section is one card of the tab.
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    group_size: int = 8
    groups_per_batch: int = 3
    max_new_tokens: int = 320  # L_max
    overlong_cache: int = 64  # L_cache: the soft penalty starts at L_max − L_cache tokens
    eps_low: float = 0.2
    eps_high: float = 0.28  # clip-higher
    iterations: int = 2  # μ
    lr: float = 3e-6
    max_grad_norm: float = 1.0
    reward_weights: tuple = (1.0, 0.5, 1.0)  # correct, boxed, overlong


# %% [reward] Rewards with overlong shaping
def last_boxed(text):
    found = re.findall(r"\\\\boxed\\{([^{}]*)\\}", text)
    return found[-1] if found else None


def overlong_penalty(length, c: Config = Config()):
    """0 up to L_max − L_cache tokens, then a linear ramp to −1 at L_max (DAPO's Eq. 13): long answers lose reward
    gradually instead of only by being cut off."""
    start = c.max_new_tokens - c.overlong_cache
    if length <= start:
        return 0.0
    if length <= c.max_new_tokens:
        return (start - length) / c.overlong_cache
    return -1.0


def rewards(text, length, gold, c: Config = Config()):
    """(correct, boxed, overlong, total): the GRPO tab's rule-based rewards plus the overlong term."""
    found = last_boxed(text)
    try:
        value = float(found.replace(",", "").replace("$", "").strip()) if found is not None else None
    except ValueError:
        value = None
    correct = 1.0 if value is not None and value == float(gold) else 0.0
    boxed = 1.0 if found is not None else 0.0
    overlong = overlong_penalty(length, c)
    w = c.reward_weights
    return correct, boxed, overlong, w[0] * correct + w[1] * boxed + w[2] * overlong


# %% [sampling] Dynamic sampling
def keep_group(correct):
    """correct: (G,) 0/1. A group that is all right or all wrong has the same correctness everywhere, so its
    advantages carry (almost) no signal: drop it and sample another problem's group instead."""
    right = int(correct.sum())
    return 0 < right < correct.numel()


def group_advantages(total):
    """As GRPO: Â_i = (r_i − mean) / (std + 1e-4) within the group."""
    return (total - total.mean()) / (total.std() + 1e-4)


# %% [objective] Clip-higher, token-level loss
def dapo_loss(logps, old_logps, advantages, mask, tokens_in_batch, c: Config = Config()):
    """Per token −min(ρ Â, clip(ρ, 1 − ε_low, 1 + ε_high) Â), summed over this micro-batch's answer tokens and divided
    by the number of answer tokens in the whole batch (not averaged per answer first). No KL term."""
    ratio = torch.exp(logps - old_logps)
    clipped = torch.clamp(ratio, 1 - c.eps_low, 1 + c.eps_high)
    a = advantages[:, None]
    per_token = -torch.min(ratio * a, clipped * a)
    return (per_token * mask).sum() / tokens_in_batch, ratio


# %% [step] One DAPO update
def answer_logps(model, prompt_ids, completion_ids, completion_mask):
    g, length = completion_ids.shape
    ids = torch.cat([prompt_ids.expand(g, -1), completion_ids], dim=1)
    mask = torch.cat([torch.ones(g, prompt_ids.shape[1], dtype=torch.long), completion_mask], dim=1)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, -length - 1 : -1]
    return logits.log_softmax(dim=-1).gather(-1, completion_ids[..., None])[..., 0]


def dapo_step(policy, optimizer, batches, c: Config = Config()):
    """batches: micro-batches of (prompt_ids, completion_ids, completion_mask, advantages, old_logps); their token-level
    losses add up to the batch's, so the gradients simply accumulate."""
    tokens_in_batch = sum(int(mask.sum()) for _, _, mask, _, _ in batches)
    optimizer.zero_grad()
    total = 0.0
    for prompt_ids, completion_ids, completion_mask, advantages, old_logps in batches:
        logps = answer_logps(policy, prompt_ids, completion_ids, completion_mask)
        loss, _ = dapo_loss(logps, old_logps, advantages, completion_mask.float(), tokens_in_batch, c)
        loss.backward()
        total += loss.detach()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
    optimizer.step()
    return total


# %% [model] One round: sample until the batch is full, then μ updates
def dapo_round(policy, optimizer, problems, sample_group, c: Config = Config(), micro=4):
    """problems: an iterator of (prompt_ids, gold). sample_group(prompt_ids) → (completion_ids, mask, texts)."""
    batches, kept = [], 0
    while kept < c.groups_per_batch:
        prompt_ids, gold = next(problems)
        completion, mask, texts = sample_group(prompt_ids)
        scored = [rewards(t, int(n), gold, c) for t, n in zip(texts, mask.sum(1))]
        if not keep_group(torch.tensor([s[0] for s in scored])):
            continue  # dynamic sampling: refill from the next problem
        advantages = group_advantages(torch.tensor([s[3] for s in scored]))
        with torch.no_grad():
            old = answer_logps(policy, prompt_ids, completion, mask)
        for i in range(0, c.group_size, micro):
            s = slice(i, i + micro)
            batches.append((prompt_ids, completion[s], mask[s], advantages[s], old[s]))
        kept += 1
    return [dapo_step(policy, optimizer, batches, c) for _ in range(c.iterations)]


# policy = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")  # no reference model: no KL term
# optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-6)
# losses = dapo_round(policy, optimizer, iter(problems), sample_group)
`;export{e as default};
