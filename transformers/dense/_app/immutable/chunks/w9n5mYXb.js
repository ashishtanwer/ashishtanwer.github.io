const e=`# Perplexity of a fixed-context language model (Qwen2.5-0.5B) on WikiText-2, with sliding windows, as plain PyTorch.
# A model that reads at most L tokens cannot score a 299,078-token text in one pass, so it reads windows of L tokens,
# one every S tokens (the stride), and scores in each window only the tokens no earlier window scored, each from the
# tokens before it in its window. Perplexity is exp of the mean negative log-likelihood over the scored tokens. Each
# "# %% [id]" section is one card of the tab.
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    context: int = 1024  # L, tokens per window
    stride: int = 512  # S, tokens between window starts; S = L: no overlap
    chunk: int = 2048  # positions per head matmul, to bound the (positions, vocab) logits


# %% [windows] The window schedule
def windows(n, context, stride):
    """(begin, end, first scored position) of every window over n tokens. A window scores the tokens after the
    previous window's end (all of them for the first window); its own first token has no context and is never
    scored, so with S = L every window loses one token."""
    out, prev_end = [], 0
    for begin in range(0, n, stride):
        end = min(begin + context, n)
        out.append((begin, end, max(prev_end, begin + 1)))
        prev_end = end
        if end == n:
            break
    return out


# %% [loss] Per-token loss
def token_loss(logits, targets):
    """logits (T, V) at the positions before each target, targets (T,) → −ln p(target), in nats:
    logsumexp over the vocabulary minus the target's logit (cross-entropy without the mean)."""
    return logits.logsumexp(-1) - logits.gather(-1, targets[:, None])[:, 0]


# %% [perplexity] Perplexity over the windows
def perplexity(losses):
    """exp(mean loss) over the scored tokens (NaN marks a token no window scored)."""
    scored = losses[~losses.isnan()]
    return math.exp(scored.double().mean().item())


# %% [model] Sliding-window evaluation
@torch.no_grad()
def evaluate(model, ids, c: Config = Config()):
    """Every token's loss under windows of c.context tokens every c.stride tokens (NaN where not scored)."""
    n = ids.numel()
    losses = torch.full((n,), float("nan"), dtype=torch.float64)
    head = model.lm_head.weight  # (V, d)
    for begin, end, first in windows(n, c.context, c.stride):
        hidden = model.model(input_ids=ids[begin:end][None]).last_hidden_state[0]  # (end − begin, d), final norm
        h = hidden[first - 1 - begin : end - 1 - begin]  # the positions that predict tokens first … end − 1
        for k in range(0, len(h), c.chunk):
            logits = (h[k : k + c.chunk] @ head.T).float()
            losses[first + k : first + k + len(logits)] = token_loss(logits, ids[first + k : first + k + len(logits)]).double()
    return losses, perplexity(losses)


# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
# ids = tokenizer("\\n\\n".join(wikitext2_test_lines), return_tensors="pt").input_ids[0]
# losses, ppl = evaluate(model, ids, Config(context=1024, stride=512))
`;export{e as default};
