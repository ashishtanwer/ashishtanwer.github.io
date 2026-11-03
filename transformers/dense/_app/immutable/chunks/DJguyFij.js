const e=`# DPO, SimPO, ORPO and KTO side by side, the Preference losses Compare tab, as plain PyTorch.
# All four learn from the same preference data without a reward model; they differ in one formula: how an answer is
# scored (against a frozen reference or not, summed or averaged over its tokens) and what the loss does with the scores.
# The tab trains each from SmolLM2-135M-Instruct on the DPO tab's eight UltraFeedback pairs, ten steps at lr 5e-6, with
# TRL's own trainers. Every loss below takes summed log-probabilities (and lengths where it averages); CPOTrainer and
# ORPOTrainer append one more end token to each answer, so their sums and lengths include it.
# Each "# %% [id]" section is one card of the tab.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    dpo_beta: float = 0.1  # DPO: scale of the log-ratio
    simpo_beta: float = 2.0  # SimPO: scale of the average log-probability
    simpo_gamma: float = 1.0  # SimPO: target margin
    orpo_lambda: float = 0.1  # ORPO: weight of the odds-ratio term (TRL calls it beta)
    kto_beta: float = 0.1  # KTO: scale of the log-ratio
    kto_lambda_d: float = 1.0  # KTO: weight of desirable answers
    kto_lambda_u: float = 1.0  # KTO: weight of undesirable answers
    lr: float = 5e-6  # the same AdamW for all four: constant, no warmup, gradient norm clipped to 1
    steps: int = 10  # every pair in every step


# %% [logps] Summed and average log-probability
def answer_logps(model, prompt_ids, answer_ids):
    """log π(y_t | x, y_<t) for every answer token; their sum log π(y | x), which DPO and KTO use, and their mean
    (1/|y|) log π(y | x), which SimPO and ORPO use. A longer answer has a lower sum whatever its quality."""
    ids = torch.cat([prompt_ids, answer_ids])[None]
    logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1]  # the positions that predict the answer
    per_token = logits.log_softmax(dim=-1).gather(-1, answer_ids[:, None])[:, 0]  # (|y|,)
    return per_token, per_token.sum(), per_token.mean()


# %% [dpo] DPO: log-ratio to a reference, summed
def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    """r = β (log π_θ(y|x) − log π_ref(y|x)); loss −log σ(r_w − r_l) per pair. A frozen copy of the starting model is
    the reference, so every margin is 0 and every pair's loss log 2 before the first update."""
    margin = beta * ((chosen - ref_chosen) - (rejected - ref_rejected))
    return -F.logsigmoid(margin), margin


# %% [simpo] SimPO: average log-probability, no reference, target margin
def simpo_loss(chosen, rejected, chosen_len, rejected_len, beta=2.0, gamma=1.0):
    """r = β log π_θ(y|x) / |y|, the policy's own per-token likelihood; loss −log σ(r_w − r_l − γ). No reference model:
    the margin is not 0 at the start, and the chosen answer has to win by γ before the loss eases off."""
    margin = beta * (chosen / chosen_len - rejected / rejected_len)
    return -F.logsigmoid(margin - gamma), margin


# %% [orpo] ORPO: SFT plus an odds ratio, no reference
def log_odds(avg_logp):
    """log [p / (1 − p)] with p = exp(average log-probability), the answer's geometric-mean token probability."""
    return avg_logp - torch.where(avg_logp < -0.6931471805599453, torch.log1p(-torch.exp(avg_logp)),
                                  torch.log(-torch.expm1(avg_logp)))


def orpo_loss(chosen, rejected, chosen_len, rejected_len, prompt, prompt_len, lam=0.1):
    """L = L_SFT + λ mean −log σ(log odds(y_w) − log odds(y_l)). L_SFT is the cross-entropy of the chosen sequences,
    averaged over every prompt and chosen-answer token of the batch (TRL's ORPOTrainer counts the prompts, all but their
    first token). The SFT term pulls the chosen answers up; the odds-ratio term separates them from the rejected ones."""
    sft = -(prompt.sum() + chosen.sum()) / (prompt_len.sum() + chosen_len.sum())
    log_or = log_odds(chosen / chosen_len) - log_odds(rejected / rejected_len)
    odds_term = -F.logsigmoid(log_or)
    return sft + lam * odds_term.mean(), sft, odds_term, log_or


# %% [kto] KTO: one answer at a time, from a reference point
def kto_loss(desirable, undesirable, ref_desirable, ref_undesirable, kl_policy, kl_ref, c: Config = Config()):
    """Each answer on its own: r = β (log π_θ − log π_ref), measured from z_ref = max(0, mean log-ratio of mismatched
    pairs, answer i − 1 after prompt i), an estimate of KL(π_θ ‖ π_ref) that carries no gradient. A desirable answer's
    value is λ_D σ(r − β z_ref), an undesirable one's λ_U σ(β z_ref − r); the loss is the mean of λ − v. Pairs are not
    needed: the tab's 16 answers are its 8 pairs taken apart."""
    z_ref = (kl_policy - kl_ref).mean().clamp(min=0).detach()
    r_d = c.kto_beta * (desirable - ref_desirable)
    r_u = c.kto_beta * (undesirable - ref_undesirable)
    loss_d = c.kto_lambda_d - c.kto_lambda_d * torch.sigmoid(r_d - c.kto_beta * z_ref)
    loss_u = c.kto_lambda_u - c.kto_lambda_u * torch.sigmoid(c.kto_beta * z_ref - r_u)
    return torch.cat([loss_d, loss_u]).mean(), z_ref, r_d - r_u


# %% [ruler] One ruler for all four
def margin_change(chosen, rejected, chosen_start, rejected_start, chosen_len=None, rejected_len=None):
    """How far a run has moved log π(y_w) − log π(y_l) since the shared starting model, in nats: summed over each answer,
    or per token when lengths are given. DPO's and KTO's margins are β times the summed change; SimPO's and ORPO's own
    rewards move with the per-token one."""
    if chosen_len is None:
        return (chosen - chosen_start) - (rejected - rejected_start)
    return (chosen - chosen_start) / chosen_len - (rejected - rejected_start) / rejected_len


# %% [greedy] Answer length
@torch.no_grad()
def greedy_answer(model, prompt_ids, max_new_tokens=160, eos=2):
    """The model's own answer to a prompt, one most likely token at a time, with the KV cache, until <|im_end|> or
    max_new_tokens: its length is what the tab's length chart counts (the end token not included)."""
    out = model(input_ids=prompt_ids[None], use_cache=True)
    cache, answer = out.past_key_values, []
    for _ in range(max_new_tokens):
        token = int(out.logits[0, -1].argmax())
        if token == eos:
            break
        answer.append(token)
        out = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return torch.tensor(answer, dtype=torch.long)


# chosen, rejected = summed log π of each pair's answers under the policy (answer_logps), ref_* under the reference
# dpo, _ = dpo_loss(chosen, rejected, ref_chosen, ref_rejected); simpo, _ = simpo_loss(chosen, rejected, lw, ll) ...
`;export{e as default};
