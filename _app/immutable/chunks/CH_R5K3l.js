const e=`# ORPO (Hong, Lee & Thorne 2024) on the base SmolLM2-135M, the ORPO tab, as plain PyTorch.
# ORPO folds preference learning into supervised fine-tuning: one loss, one model, no reference model and no separate
# SFT stage. The SFT term teaches the chosen sequence token by token (TRL's trainer counts the prompt's tokens too);
# the odds-ratio term pushes the odds of the chosen answer above the odds of the rejected one, where odds = p / (1 − p)
# and p is the answer's length-normalized (geometric-mean) probability. The tab's run is TRL's ORPOTrainer; each
# "# %% [id]" section is one card of the tab.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    lam: float = 0.1  # λ, the weight of the odds-ratio term (TRL calls it beta)
    lr: float = 8e-6  # AdamW, constant, no warmup
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    steps: int = 10


# %% [logps] Per-token log-probabilities of a sequence
def sequence_logps(model, prompt_ids, answer_ids):
    """log-probabilities of every token after the first: (the prompt's, the answer's). The odds use the answer's
    mean; the SFT term uses both."""
    ids = torch.cat([prompt_ids, answer_ids])
    logits = model(input_ids=ids[None]).logits[0, :-1]  # position t predicts token t + 1
    per_token = logits.log_softmax(dim=-1).gather(-1, ids[1:, None])[:, 0]
    n = len(prompt_ids) - 1
    return per_token[:n], per_token[n:]  # (n − 1,), (|y|,)


# %% [odds] Log odds and the odds ratio
def log1mexp(x):
    """log(1 − e^x) for x < 0, accurate on both sides of x = −log 2."""
    return torch.where(x < -0.6931471805599453, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


def log_odds(avg_logp):
    """log odds(y|x) = log p − log(1 − p), with log p the answer's average per-token log-probability."""
    return avg_logp - log1mexp(avg_logp)


def log_odds_ratio(avg_chosen, avg_rejected):
    """log [odds(y_w|x) / odds(y_l|x)]: positive once the chosen answer is the more likely of the two."""
    return log_odds(avg_chosen) - log_odds(avg_rejected)


# %% [loss] ORPO loss
def orpo_loss(prompt_token_logps, chosen_token_logps, rejected_token_logps, lam=0.1):
    """L = L_SFT + λ · mean over pairs of −log σ(log odds ratio). L_SFT is the cross-entropy of the prompts and chosen
    answers, averaged over all their tokens in the batch (as TRL computes it). Lists of per-token log-probabilities,
    one per pair."""
    sft = -torch.cat([*prompt_token_logps, *chosen_token_logps]).mean()
    avg_chosen = torch.stack([t.mean() for t in chosen_token_logps])
    avg_rejected = torch.stack([t.mean() for t in rejected_token_logps])
    odds_term = -F.logsigmoid(log_odds_ratio(avg_chosen, avg_rejected))
    return sft + lam * odds_term.mean(), sft, odds_term


# %% [step] One ORPO update
def orpo_step(model, optimizer, pairs, c: Config = Config()):
    """One step on a batch of (prompt_ids, chosen_ids, rejected_ids): both terms from one forward per answer."""
    scored = [sequence_logps(model, x, y_w) for x, y_w, _ in pairs]
    rejected = [sequence_logps(model, x, y_l)[1] for x, _, y_l in pairs]
    loss, sft, odds_term = orpo_loss([p for p, _ in scored], [a for _, a in scored], rejected, c.lam)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
    optimizer.step()
    return loss.detach(), sft.detach(), odds_term.detach()


# %% [model] The whole run
def train(model, pairs, c: Config = Config()):
    """From the base model straight to preference-tuned: no SFT stage first, no reference copy."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=c.betas, eps=c.eps, weight_decay=c.weight_decay)
    return [orpo_step(model, optimizer, pairs, c) for _ in range(c.steps)]


# model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")  # the base model
# pairs = [(prompt_ids, chosen_ids, rejected_ids), ...]  # the DPO tab's pairs in SmolLM2-Instruct's chat template
# history = train(model, pairs)
`;export{e as default};
