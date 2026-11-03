const e=`# Warmup–stable–decay (WSD), as the Pretraining Explainer trains its small GPT with it (4 layers, d_model 256,
# 4 heads, context 256, ≈16.1M parameters, AdamW on TinyStories): a short linear warmup, a long stretch at the peak
# learning rate, and a short decay at the end. Cosine decay has to know the run's length from the first step; WSD
# keeps the learning rate flat until the decay, so any checkpoint of the flat stretch can be decayed into a finished
# model. Every run of the tab (scripts/pretraining/recipes.py) takes its learning rate from these schedules and
# branches its decays off the trunk the way branch() does.
import math
from dataclasses import dataclass

import torch.nn.functional as F


@dataclass
class Config:
    peak_lr: float = 4e-3  # the AdamW tab's peak learning rate, the best of its 500-step sweep
    warmup: int = 100  # steps of linear warmup, the same for every schedule
    steps: int = 2000  # the run's length in optimizer steps (32 × 256 tokens each)
    final_lr_fraction: float = 0.1  # both schedules end at 10 % of the peak
    decay_fraction: float = 0.2  # WSD decays over the last 20 % of the run


# %% [warmup] Linear warmup
def warmup_lr(step, c: Config):
    """Step 0 trains at peak / warmup, step warmup − 1 at the peak: the moments of AdamW start at zero and the
    first gradients are large, so the first steps are kept small."""
    return c.peak_lr * (step + 1) / c.warmup


# %% [cosine] Cosine decay
def cosine_lr(step, steps, c: Config):
    """After the warmup, half a cosine from the peak down to final_lr_fraction · peak at the last step, steps − 1.
    The curve depends on \`steps\` from the very first step after warmup: the run's length is fixed in advance."""
    if step < c.warmup:
        return warmup_lr(step, c)
    t = (step - c.warmup) / max(1, steps - 1 - c.warmup)  # 0 … 1
    return c.peak_lr * (c.final_lr_fraction + (1 - c.final_lr_fraction) * 0.5 * (1 + math.cos(math.pi * t)))


# %% [decay-shapes] Decay shapes
def decay_factor(progress, shape):
    """How much of the way from the peak to the final learning rate is left, for progress 0 … 1 of the decay."""
    if shape == "linear":
        return 1 - progress
    if shape == "1-sqrt":  # falls fast at first and slowly at the end (Hägele et al. 2024)
        return 1 - math.sqrt(progress)
    if shape == "cosine":
        return 0.5 * (1 + math.cos(math.pi * progress))
    raise ValueError(shape)


# %% [wsd] Warmup–stable–decay
def decay_start(steps, c: Config):
    """The first step of the decay: the last decay_fraction of the run."""
    return steps - round(c.decay_fraction * steps)


def wsd_lr(step, steps, c: Config, shape="linear"):
    """Warmup, then the peak until decay_start, then down to final_lr_fraction · peak at the last step. Until
    decay_start the schedule does not depend on \`steps\`, so one constant run serves every length."""
    if step < c.warmup:
        return warmup_lr(step, c)
    start = decay_start(steps, c)
    if step < start:
        return c.peak_lr
    progress = (step - start) / max(1, steps - 1 - start)  # 0 at the first decay step, 1 at the last step
    return c.peak_lr * (c.final_lr_fraction + (1 - c.final_lr_fraction) * decay_factor(progress, shape))


def constant_lr(step, c: Config):
    """Warmup and then the peak for ever: the stable phase on its own, the trunk the decays branch from."""
    return warmup_lr(step, c) if step < c.warmup else c.peak_lr


# %% [train] The training loop
def train(model, optimizer, batches, lr_at, start, stop, on_step=None):
    """Steps start … stop − 1 of a run: set the learning rate the schedule gives for this step, take one AdamW step on
    the next batch. \`batches\` yields the same batches in the same order for every run with the seed."""
    for step in range(start, stop):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step) * group.get("lr_mult", 1.0)
        x, y = batches.next()[:2]
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if on_step:
            on_step(step, loss.detach())


# %% [branch] Branching a decay off the trunk
def checkpoint(model, optimizer, batches, step):
    """Everything a run needs to carry on exactly: the weights, AdamW's moments and step counts, and where the batch
    stream is."""
    return {
        "step": step,
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "batches": batches.generator.get_state(),
    }


def branch(model, optimizer, batches, saved, steps, c: Config, shape="linear", on_step=None):
    """Load a checkpoint of the constant trunk taken at decay_start(steps) and train the remaining steps with the
    decay: the same run as a WSD schedule of \`steps\` steps from the start, at the cost of only its decay."""
    assert saved["step"] == decay_start(steps, c), "branch from the step where this length's decay begins"
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    batches.generator.set_state(saved["batches"])
    train(model, optimizer, batches, lambda s: wsd_lr(s, steps, c, shape), saved["step"], steps, on_step)
    return model
`;export{e as default};
