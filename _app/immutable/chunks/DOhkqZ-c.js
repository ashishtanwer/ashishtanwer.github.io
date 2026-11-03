const e=`# Continual pretraining, as the Pretraining Explainer runs it: the shared small GPT (4 layers, d_model 256, 4 heads,
# context 256, ≈16.1M parameters) is pretrained on TinyStories (domain A) with warmup and cosine decay, then trained
# on Wikipedia (wikitext-103, domain B) from its last checkpoint. Two knobs decide how much of A it keeps: the learning
# rate after the switch (stay at the low final rate, or warm it back up to the peak and decay again) and replay
# (a fixed share of every batch drawn from A again). scripts/pretraining/continual_runs.py trains with these blocks.
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    peak_lr: float = 4e-3  # the AdamW tab's peak, also the first phase's
    final_lr_fraction: float = 0.1  # the first phase ended at 0.1 × peak
    rewarm_steps: int = 100  # steps to climb back
    rewarm_to: float = 1.0  # … to this fraction of the peak
    continue_steps: int = 1000  # steps after the switch
    batch: int = 32  # windows of 256 tokens a step
    replay: float = 0.0  # share of each batch drawn from domain A
    clip: float = 1.0  # gradient norm clipped to this in both phases


# %% [replay] Replay
def replay_rows(c: Config):
    """How many of a batch's windows come from A and how many from B; every step has the same split."""
    a = round(c.replay * c.batch)
    return {"stories": a, "wiki": c.batch - a}


# %% [rewarm] Learning rate after the switch
def constant_lr(t, c: Config):
    """No rewarming: carry on at the rate the first phase ended at."""
    return c.final_lr_fraction * c.peak_lr


def rewarm_lr(t, c: Config):
    """Rewarming: from the final rate back up to rewarm_to · peak over rewarm_steps, then half a cosine down to the
    final rate at the last step, the first phase's schedule laid over the new steps. t counts steps since the switch."""
    low, high = c.final_lr_fraction * c.peak_lr, c.rewarm_to * c.peak_lr
    if t < c.rewarm_steps:
        return low + (high - low) * (t + 1) / c.rewarm_steps
    u = (t - c.rewarm_steps) / max(1, c.continue_steps - 1 - c.rewarm_steps)
    return low + (high - low) * 0.5 * (1 + math.cos(math.pi * u))


# %% [train-step] One training step
def train_step(model, x, y, optimizer, c: Config = Config()):
    """One AdamW step on the mixed batch (A's windows first, then B's); the loss is their mean, and the gradient is
    scaled down to norm c.clip when it is larger. Returns the loss and the gradient's norm before clipping."""
    logits = model(x)
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
    optimizer.step()
    return loss.detach(), norm


# %% [forgetting] Measuring forgetting
@torch.no_grad()
def token_losses(model, x, y):
    """Each position's next-token loss (nats) on fixed windows; the page compares them before and after the switch."""
    logits = model(x)
    return F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view(y.shape)


def forgetting(losses_a, switch_index):
    """How far domain A's validation loss has risen above its value at the switch, at every evaluation after it."""
    base = losses_a[switch_index]
    return [v - base for v in losses_a[switch_index:]]
`;export{e as default};
