const e=`# KTO (Ethayarajh et al. 2024) on SmolLM2-135M-Instruct, the KTO tab, as plain PyTorch.
# KTO learns from single answers marked desirable or undesirable (a thumbs up or down), with no pairs. Like DPO it scores
# an answer by its implied reward β log π_θ(y|x)/π_ref(y|x), but it measures that reward from a reference point z_ref,
# a batch estimate of KL(π_θ ‖ π_ref) made from answers paired with the wrong prompt, through a sigmoid value function
# borrowed from Kahneman and Tversky's prospect theory. The tab's run is TRL's KTOTrainer with loss_type="kto".
# Each "# %% [id]" section is one card of the tab.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    beta: float = 0.1  # reward scale
    lambda_d: float = 1.0  # weight of desirable rows (TRL: desirable_weight)
    lambda_u: float = 1.0  # weight of undesirable rows (TRL: undesirable_weight)
    lr: float = 5e-6  # AdamW, constant, no warmup
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    steps: int = 10  # each step uses the same 16 rows


# %% [logps] Log-probability of an answer
def answer_logp(model, prompt_ids, answer_ids):
    """log π(y|x) = Σ_t log π(y_t | x, y_<t): the per-token log-probabilities of the answer, summed."""
    ids = torch.cat([prompt_ids, answer_ids])[None]
    logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1]  # the positions that predict the answer
    per_token = logits.log_softmax(dim=-1).gather(-1, answer_ids[:, None])[:, 0]  # (|y|,)
    return per_token.sum(), per_token


# %% [reward] Implied reward
def implied_rewards(policy_logps, ref_logps, beta=0.1):
    """r(x, y) = β (log π_θ(y|x) − log π_ref(y|x)), one per row. 0 at the start, when the policy is the reference."""
    return beta * (policy_logps - ref_logps)


# %% [kl] Reference point from mismatched pairs
def mismatched(answers):
    """The KL pairs of a batch: answer i − 1 after prompt i (the batch rotated by one), so no answer meets its own
    prompt as long as the batch holds no two rows with the same prompt next to each other."""
    return [answers[-1]] + answers[:-1]


def reference_point(policy_kl_logps, ref_kl_logps):
    """z_ref = max(0, mean over the batch of log π_θ(y'|x) − log π_ref(y'|x)) on the mismatched pairs (x, y'): an
    estimate of KL(π_θ ‖ π_ref) in nats. It carries no gradient (the policy scores the pairs under no_grad)."""
    return (policy_kl_logps - ref_kl_logps).mean().clamp(min=0).detach()


# %% [value] Value function
def kto_value(rewards, z_ref, desirable, c: Config = Config()):
    """v(x, y) = λ_D σ(r − β z_ref) for a desirable row, λ_U σ(β z_ref − r) for an undesirable one. Each row gains
    value as its reward moves past the reference point in its own direction; the sigmoid saturates on both sides."""
    sign = desirable.to(rewards.dtype) * 2 - 1
    weight = torch.where(desirable, c.lambda_d, c.lambda_u)
    return weight * torch.sigmoid(sign * (rewards - c.beta * z_ref))


# %% [loss] KTO loss
def kto_loss(policy_logps, ref_logps, policy_kl_logps, ref_kl_logps, desirable, c: Config = Config()):
    """mean over the rows of λ_y − v(x, y): ½ λ per row at the start, never more than λ, never below 0."""
    rewards = implied_rewards(policy_logps, ref_logps, c.beta)
    z_ref = reference_point(policy_kl_logps, ref_kl_logps)
    weight = torch.where(desirable, c.lambda_d, c.lambda_u)
    return (weight - kto_value(rewards, z_ref, desirable, c)).mean()


# %% [step] One KTO update
def kto_step(policy, reference, optimizer, rows, c: Config = Config()):
    """One step on a batch of (prompt_ids, answer_ids, desirable) rows: score every answer with the policy (with
    gradient) and the reference, score the mismatched pairs with both (no gradient), the loss, backward, clip, AdamW."""
    prompts = [x for x, _, _ in rows]
    answers = [y for _, y, _ in rows]
    desirable = torch.tensor([d for _, _, d in rows])
    policy_logps = torch.stack([answer_logp(policy, x, y)[0] for x, y in zip(prompts, answers)])
    with torch.no_grad():
        ref_logps = torch.stack([answer_logp(reference, x, y)[0] for x, y in zip(prompts, answers)])
        pairs = list(zip(prompts, mismatched(answers)))
        policy_kl = torch.stack([answer_logp(policy, x, y)[0] for x, y in pairs])
        ref_kl = torch.stack([answer_logp(reference, x, y)[0] for x, y in pairs])
    loss = kto_loss(policy_logps, ref_logps, policy_kl, ref_kl, desirable, c)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
    optimizer.step()
    return loss.detach()


# %% [model] The whole run
def train(model, reference, rows, c: Config = Config()):
    """A frozen copy of the starting model is the reference. Each row needs its own answer and one mismatched answer
    scored by both models: four forward passes per row, one of them with gradient."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=c.betas, eps=c.eps, weight_decay=c.weight_decay)
    return [kto_step(model, reference, optimizer, rows, c) for _ in range(c.steps)]


# model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct"); reference = a frozen copy
# rows = [(prompt_ids, answer_ids, True), ...]  # the 8 desirable kto-mix-14k rows, then the 8 undesirable ones
# losses = train(model, reference, rows)
`;export{e as default};
