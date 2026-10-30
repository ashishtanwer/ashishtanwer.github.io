const e=`# DPO (Rafailov et al. 2023) on SmolLM2-135M-Instruct, the DPO tab, as plain PyTorch.
# A preference pair is one prompt x with a chosen answer y_w and a rejected answer y_l. DPO needs no reward model and
# no sampling: it scores both answers with the policy π_θ and with a frozen copy π_ref, and pushes the policy's
# log-ratio log π_θ(y|x) − log π_ref(y|x) up for y_w and down for y_l through a logistic loss.
# The tab's run is TRL's DPOTrainer with these settings; each "# %% [id]" section is one card of the tab.
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    beta: float = 0.1  # how far the policy may move from the reference; the implicit reward is β · log-ratio
    lr: float = 5e-6  # AdamW, constant, no warmup
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # Trainer's gradient clipping
    steps: int = 10  # each step uses the same 8 pairs


# %% [logps] Log-probability of an answer
def answer_logps(model, prompt_ids, answer_ids):
    """log π(y_t | x, y_<t) for every answer token, and their sum log π(y | x). Prompt tokens are not scored.

    prompt_ids, answer_ids: (n,) and (m,) token ids of the chat-templated prompt and of the answer (with <|im_end|>)."""
    ids = torch.cat([prompt_ids, answer_ids])[None]  # (1, n + m)
    logits = model(input_ids=ids).logits[0]  # (n + m, vocab)
    predicting = logits[len(prompt_ids) - 1 : -1]  # position t predicts token t + 1: the m answer positions
    per_token = predicting.log_softmax(dim=-1).gather(-1, answer_ids[:, None])[:, 0]  # (m,)
    return per_token, per_token.sum()


# %% [reward] Implicit reward
def implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """r(x, y) = β · (log π_θ(y|x) − log π_ref(y|x)): how much more likely the policy makes an answer than the
    reference did, in nats, scaled by β. Both are 0 before the first update, when π_θ = π_ref."""
    chosen_reward = beta * (policy_chosen - ref_chosen)
    rejected_reward = beta * (policy_rejected - ref_rejected)
    return chosen_reward, rejected_reward, chosen_reward - rejected_reward  # the last is the margin


# %% [loss] DPO loss
def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """−log σ(r_w − r_l), averaged over the pairs: the Bradley–Terry likelihood that the chosen answer wins, with the
    implicit rewards in place of a reward model's scores. It starts at log 2 ≈ 0.693 (margin 0)."""
    _, _, margin = implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return -F.logsigmoid(margin).mean()


# %% [gradient] Gradient on the two answers
def answer_gradients(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """∂L/∂log π_θ(y_w) and ∂L/∂log π_θ(y_l) for each pair (L averaged over the pairs). A pair whose margin is
    still small or wrong gets weight σ(−margin) close to 1; one the policy already ranks well gets less. The
    sum's gradient spreads equally over the answer's tokens: each token's log-probability gets the same push."""
    _, _, margin = implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    weight = torch.sigmoid(-margin)  # (pairs,)
    n = margin.numel()
    return -beta * weight / n, beta * weight / n  # down the loss = up for chosen, down for rejected


# %% [step] One DPO update
def dpo_step(policy, reference, optimizer, pairs, c: Config = Config()):
    """One step on a batch of pairs: score each answer with the policy (with gradients) and the frozen reference,
    take the DPO loss, backpropagate, clip the gradient norm to 1 and let AdamW move the weights.

    pairs: list of (prompt_ids, chosen_ids, rejected_ids)."""
    scores = {"pc": [], "pr": [], "rc": [], "rr": []}
    for prompt_ids, chosen_ids, rejected_ids in pairs:
        scores["pc"].append(answer_logps(policy, prompt_ids, chosen_ids)[1])
        scores["pr"].append(answer_logps(policy, prompt_ids, rejected_ids)[1])
        with torch.no_grad():
            scores["rc"].append(answer_logps(reference, prompt_ids, chosen_ids)[1])
            scores["rr"].append(answer_logps(reference, prompt_ids, rejected_ids)[1])
    pc, pr, rc, rr = (torch.stack(scores[k]) for k in ("pc", "pr", "rc", "rr"))
    loss = dpo_loss(pc, pr, rc, rr, c.beta)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
    optimizer.step()
    return loss.detach(), grad_norm


# %% [model] The whole run
def train(model, pairs, c: Config = Config()):
    """The reference is a frozen copy of the starting model; only the policy is trained."""
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=c.betas, eps=c.eps, weight_decay=c.weight_decay)
    return [dpo_step(model, reference, optimizer, pairs, c) for _ in range(c.steps)]


# model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
# pairs = [(prompt_ids, chosen_ids, rejected_ids), ...]  # 8 UltraFeedback pairs, chat-templated
# losses = train(model, pairs)
`;export{e as default};
