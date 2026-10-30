const e=`# SimPO (Meng, Xia & Chen 2024) on SmolLM2-135M-Instruct, the SimPO tab, as plain PyTorch.
# Like DPO it learns from preference pairs (a prompt x, a chosen answer y_w, a rejected answer y_l), but it needs no
# reference model: an answer's reward is β times its average per-token log-probability under the policy itself, and the
# chosen answer must beat the rejected one by a target margin γ. Averaging (not summing) keeps long answers from being
# penalised just for having more tokens. The tab's run is TRL's CPOTrainer with loss_type="simpo" and cpo_alpha=0.
# Each "# %% [id]" section is one card of the tab.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    beta: float = 2.0  # reward scale
    gamma: float = 1.0  # target reward margin (γ / β = 0.5)
    lr: float = 5e-6  # AdamW, constant, no warmup
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    steps: int = 10  # each step uses the same 8 pairs


# %% [logps] Average log-probability of an answer
def average_logp(model, prompt_ids, answer_ids):
    """(1/|y|) Σ_t log π(y_t | x, y_<t): the per-token log-probabilities of the answer, averaged over its |y| tokens.
    exp of it is the geometric mean of the token probabilities."""
    ids = torch.cat([prompt_ids, answer_ids])[None]
    logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1]  # the positions that predict the answer
    per_token = logits.log_softmax(dim=-1).gather(-1, answer_ids[:, None])[:, 0]  # (|y|,)
    return per_token.mean(), per_token


# %% [reward] Length-normalized reward
def simpo_rewards(avg_chosen, avg_rejected, beta=2.0):
    """r(x, y) = (β / |y|) log π_θ(y|x) = β × the average log-probability. No π_ref: the policy's own likelihood is the
    reward, so rewards are negative and do not start at 0 the way DPO's implicit rewards do."""
    chosen, rejected = beta * avg_chosen, beta * avg_rejected
    return chosen, rejected, chosen - rejected


# %% [loss] SimPO loss
def simpo_loss(avg_chosen, avg_rejected, beta=2.0, gamma=1.0):
    """−log σ(r_w − r_l − γ), averaged over the pairs: the chosen answer has to win by at least γ before the loss
    flattens out."""
    _, _, margin = simpo_rewards(avg_chosen, avg_rejected, beta)
    return -F.logsigmoid(margin - gamma).mean()


# %% [gradient] Gradient on the two answers
def answer_gradients(avg_chosen, avg_rejected, beta=2.0, gamma=1.0):
    """∂L/∂(average log-probability) of the chosen and the rejected answer of each pair. Every token of an answer gets
    the same push divided by |y|: a token of a long answer moves less than a token of a short one."""
    _, _, margin = simpo_rewards(avg_chosen, avg_rejected, beta)
    weight = torch.sigmoid(-(margin - gamma)) / margin.numel()
    return -beta * weight, beta * weight


# %% [step] One SimPO update
def simpo_step(policy, optimizer, pairs, c: Config = Config()):
    """One step on a batch of (prompt_ids, chosen_ids, rejected_ids): average log-probabilities, the loss, backward,
    clip the gradient norm, AdamW."""
    chosen = torch.stack([average_logp(policy, x, y_w)[0] for x, y_w, _ in pairs])
    rejected = torch.stack([average_logp(policy, x, y_l)[0] for x, _, y_l in pairs])
    loss = simpo_loss(chosen, rejected, c.beta, c.gamma)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
    optimizer.step()
    return loss.detach()


# %% [model] The whole run
def train(model, pairs, c: Config = Config()):
    """Only the policy: SimPO keeps no reference copy, so it needs half of DPO's forward passes per step."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=c.betas, eps=c.eps, weight_decay=c.weight_decay)
    return [simpo_step(model, optimizer, pairs, c) for _ in range(c.steps)]


# model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
# pairs = [(prompt_ids, chosen_ids, rejected_ids), ...]  # the DPO tab's 8 UltraFeedback pairs, chat-templated
# losses = train(model, pairs)
`;export{e as default};
