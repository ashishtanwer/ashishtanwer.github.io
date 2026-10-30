const e=`# GRPO (Shao et al. 2024, DeepSeekMath; the RL of DeepSeek-R1) on Qwen2.5-0.5B-Instruct and GSM8K, as plain PyTorch.
# No reward model and no value model: the policy samples a group of G answers to one problem, a rule scores each one
# (is the boxed number right? is there a boxed number?), and each answer's advantage is its reward relative to the
# rest of its group. The update is PPO's clipped ratio objective with those advantages on every token, plus a KL
# penalty that keeps the policy near a frozen reference. The tab's run uses TRL GRPOTrainer's settings below.
# Each "# %% [id]" section is one card of the tab.
import re
from dataclasses import dataclass

import torch


@dataclass
class Config:
    group_size: int = 8  # G answers per problem
    max_new_tokens: int = 320
    temperature: float = 1.0  # sample from the model's own distribution: no top-k, no top-p
    beta: float = 0.04  # KL coefficient
    epsilon: float = 0.2  # clip range of the ratio π_θ / π_old
    iterations: int = 2  # μ: updates on each sampled batch
    lr: float = 3e-6  # AdamW, constant
    max_grad_norm: float = 1.0
    reward_weights: tuple = (1.0, 0.5)  # correct, boxed


# %% [reward] Rule-based rewards
def last_boxed(text):
    found = re.findall(r"\\\\boxed\\{([^{}]*)\\}", text)
    return found[-1] if found else None


def rewards(text, gold, c: Config = Config()):
    """(correct, boxed, total) for one answer: 1 if the last \\\\boxed{} holds the reference number, 1 if there is a
    \\\\boxed{} at all; the total weights them 1 and 0.5. An answer cut off at the length limit usually gets 0."""
    found = last_boxed(text)
    try:
        value = float(found.replace(",", "").replace("$", "").strip()) if found is not None else None
    except ValueError:
        value = None
    correct = 1.0 if value is not None and value == float(gold) else 0.0
    boxed = 1.0 if found is not None else 0.0
    return correct, boxed, c.reward_weights[0] * correct + c.reward_weights[1] * boxed


# %% [advantage] Group-relative advantage
def group_advantages(total):
    """total: (G,) rewards of one problem's group. Â_i = (r_i − mean) / (std + 1e-4), std with Bessel's correction.
    The group's own mean is the baseline a value model would otherwise learn; if every answer got the same reward,
    all advantages are 0 and the group teaches nothing."""
    return (total - total.mean()) / (total.std() + 1e-4)


# %% [logps] Per-token log-probabilities
def answer_logps(model, prompt_ids, completion_ids, completion_mask):
    """log π(o_t | q, o_<t) at every answer position. prompt_ids: (1, n), shared by the group; completion_ids and
    completion_mask: (G, L), right-padded after each answer's end-of-turn token."""
    g, length = completion_ids.shape
    ids = torch.cat([prompt_ids.expand(g, -1), completion_ids], dim=1)
    mask = torch.cat([torch.ones(g, prompt_ids.shape[1], dtype=torch.long), completion_mask], dim=1)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, -length - 1 : -1]  # positions that predict o_t
    return logits.log_softmax(dim=-1).gather(-1, completion_ids[..., None])[..., 0]  # (G, L)


# %% [objective] Clipped objective with KL
def grpo_loss(logps, old_logps, ref_logps, advantages, mask, c: Config = Config()):
    """Per token: −min(ρ Â, clip(ρ, 1 − ε, 1 + ε) Â) + β KL, with ρ = π_θ / π_old and the KL estimate
    π_ref/π_θ − log(π_ref/π_θ) − 1 (≥ 0). Averaged over each answer's tokens, then over the answers.
    On the first update of a batch π_θ = π_old, so ρ = 1 and nothing is clipped."""
    ratio = torch.exp(logps - old_logps)
    clipped = torch.clamp(ratio, 1 - c.epsilon, 1 + c.epsilon)
    a = advantages[:, None]
    per_token = -torch.min(ratio * a, clipped * a)
    kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1
    per_token = per_token + c.beta * kl
    per_answer = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    return per_answer.mean(), ratio, kl


# %% [step] One GRPO update
def grpo_step(policy, optimizer, batches, c: Config = Config()):
    """One AdamW step over the sampled batch, given as micro-batches of (prompt_ids, completion_ids, completion_mask,
    advantages, old_logps, ref_logps); their losses are averaged by accumulating gradients."""
    optimizer.zero_grad()
    total = 0.0
    for prompt_ids, completion_ids, completion_mask, advantages, old_logps, ref_logps in batches:
        logps = answer_logps(policy, prompt_ids, completion_ids, completion_mask)
        loss, _, _ = grpo_loss(logps, old_logps, ref_logps, advantages, completion_mask.float(), c)
        (loss / len(batches)).backward()
        total += loss.detach() / len(batches)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
    optimizer.step()
    return total


# %% [model] One round: sample, score, update μ times
@torch.no_grad()
def sample_group(policy, prompt_ids, eos_ids, pad_id, c: Config = Config()):
    out = policy.generate(prompt_ids.expand(c.group_size, -1), do_sample=True, temperature=c.temperature, top_k=0,
                          top_p=1.0, max_new_tokens=c.max_new_tokens, eos_token_id=eos_ids, pad_token_id=pad_id)
    completion = out[:, prompt_ids.shape[1]:]
    is_eos = torch.isin(completion, torch.tensor(eos_ids))
    end = torch.where(is_eos.any(1), is_eos.int().argmax(1), completion.shape[1])  # first eos, kept in the answer
    mask = (torch.arange(completion.shape[1])[None] <= end[:, None]).long()
    return completion, mask


def grpo_round(policy, reference, optimizer, problems, tokenizer, eos_ids, c: Config = Config(), micro=4):
    """problems: list of (prompt_ids (1, n), gold answer). Every answer of every group is used μ times."""
    batches = []
    for prompt_ids, gold in problems:
        completion, mask = sample_group(policy, prompt_ids, eos_ids, tokenizer.pad_token_id, c)
        texts = tokenizer.batch_decode(completion.masked_fill(mask == 0, tokenizer.pad_token_id),
                                       skip_special_tokens=True)
        total = torch.tensor([rewards(t, gold, c)[2] for t in texts])
        advantages = group_advantages(total)
        with torch.no_grad():
            old = answer_logps(policy, prompt_ids, completion, mask)  # π_old: the policy that sampled
            ref = answer_logps(reference, prompt_ids, completion, mask)
        for i in range(0, c.group_size, micro):
            s = slice(i, i + micro)
            batches.append((prompt_ids, completion[s], mask[s], advantages[s], old[s], ref[s]))
    return [grpo_step(policy, optimizer, batches, c) for _ in range(c.iterations)]


# policy = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"); reference = a frozen copy
# optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-6)
# losses = grpo_round(policy, reference, optimizer, problems, tokenizer, eos_ids=[151645, 151643])
`;export{e as default};
