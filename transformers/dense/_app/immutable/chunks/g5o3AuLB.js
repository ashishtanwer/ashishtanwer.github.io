const e=`# A process reward model (Lightman et al. 2023, "Let's Verify Step by Step"), Skywork-o1-Open-PRM-Qwen-2.5-1.5B, as
# plain PyTorch around transformers' Qwen2Model: instead of one number for a whole answer, the model gives a number
# after every step, its probability that the solution is still on track. Steps are lines; the score of a step is read
# at the newline token that ends it. The step scores are aggregated into one score per answer (min, last, product,
# mean) and used for best-of-N, against the outcome reward model. Follows Skywork's own inference code
# (github.com/SkyworkAI/skywork-o1-prm-inference @ 719b56b); the checkpoint's remote code is not needed.
# Each "# %% [id]" section is one card of the tab.
import json
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    hidden_size: int = 1536  # Qwen2.5-Math-1.5B, which the PRM was fine-tuned from
    step_token: str = "\\n"
    threshold: float = 0.5  # a step scored below this is flagged


# %% [inputs] Steps and where they are scored
def prm_inputs(tokenizer, problem, answer, c: Config = Config()):
    """'<|im_start|>' + problem + '\\\\n', then the answer cut at every newline: each piece is tokenized on its own and
    followed by the newline token, whose position is that piece's score. A blank line is an empty piece (a lone
    newline). Trailing newlines are dropped first, so a partial answer is scored at its last written line.
    Returns the ids, the positions of the newlines, and the pieces."""
    ids = tokenizer.encode(tokenizer.bos_token + problem + "\\n")
    newline = tokenizer.encode(c.step_token)[-1]
    flags, pieces = [], answer.rstrip("\\n").split(c.step_token)
    for piece in pieces:
        ids = ids + (tokenizer.encode(piece) if piece else []) + [newline]
        flags.append(len(ids) - 1)
    return ids, flags, pieces


# %% [value-head] The value head
class ProcessRewardModel(torch.nn.Module):
    """The decoder's last hidden state at every position through one linear layer and a sigmoid: (B, T) probabilities.
    Only the values at the newlines are read."""

    def __init__(self, backbone, c: Config = Config()):
        super().__init__()
        self.backbone = backbone  # transformers' Qwen2Model
        self.v_head = torch.nn.Linear(c.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, H), normed
        return torch.sigmoid(self.v_head(h)[..., 0])

    @torch.no_grad()
    def step_scores(self, tokenizer, problem, answer):
        ids, flags, pieces = prm_inputs(tokenizer, problem, answer)
        values = self(torch.tensor([ids]))[0]
        return values[flags], pieces  # one score per piece


def steps(scores, pieces):
    """The steps: the non-empty pieces and their scores."""
    keep = [i for i, piece in enumerate(pieces) if piece.strip()]
    return scores[keep], [pieces[i] for i in keep]


# %% [flag] The first flagged step
def first_flag(step_scores, c: Config = Config()):
    """Index of the first step scored under the threshold, or −1: where the PRM says the solution went wrong."""
    low = (step_scores < c.threshold).nonzero()
    return int(low[0]) if len(low) else -1


# %% [aggregate] From step scores to one score
def aggregate(step_scores):
    return {
        "min": step_scores.min(),  # as good as its weakest step
        "last": step_scores[-1],  # the PRM's estimate after the whole solution
        "product": step_scores.prod(),  # every step right, if the steps were independent
        "mean": step_scores.mean(),  # Skywork's own weighting for best-of-N
    }


# %% [rank] Ranking right above wrong, and best-of-N
def pairwise_accuracy(scores, right):
    """scores, right: (N,) for one problem. Share of (right, wrong) answer pairs in which the right one scores higher
    (ties count half); this is what decides whether best-of-N picks a right answer."""
    r, w = scores[right.bool()], scores[~right.bool()]
    if not len(r) or not len(w):
        return None, 0
    diff = r[:, None] - w[None, :]
    return float(((diff > 0).double() + 0.5 * (diff == 0).double()).mean()), diff.numel()


def best_of_n_accuracy(scores, right, n):
    """Chance that the highest of n drawn scores belongs to a right answer: the r-th lowest score (from 0) is the
    best of C(r, n − 1) of the C(N, n) subsets."""
    order = torch.argsort(scores).tolist()
    return sum(float(right[i]) * math.comb(r, n - 1) for r, i in enumerate(order)) / math.comb(len(scores), n)


# %% [model] Loading the checkpoint and scoring answers
def load(path, c: Config = Config()):
    """The checkpoint is a stock Qwen2 (model.*), its LM head (unused) and the value head (v_head.summary.*)."""
    from transformers import Qwen2Config, Qwen2Model

    config = json.loads((path / "config.json").read_text())
    for key in ("auto_map", "architectures"):  # point at the remote code, which is not used
        config.pop(key)
    model = ProcessRewardModel(Qwen2Model(Qwen2Config(**config)), c)
    state = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.backbone.load_state_dict({k[len("model."):]: v for k, v in state.items() if k.startswith("model.")})
    model.v_head.load_state_dict({k[len("v_head.summary."):]: v for k, v in state.items() if k.startswith("v_head.")})
    return model.float().eval()


@torch.no_grad()
def score_answers(model, tokenizer, problem, answers, pad_id):
    """Several answers in one right-padded batch (padding after a sequence does not change a causal model's values at
    its real positions): every piece's score, per answer."""
    inputs = [prm_inputs(tokenizer, problem, a) for a in answers]
    width = max(len(ids) for ids, _, _ in inputs)
    ids = torch.full((len(inputs), width), pad_id)
    mask = torch.zeros((len(inputs), width), dtype=torch.long)
    for i, (row, _, _) in enumerate(inputs):
        ids[i, : len(row)], mask[i, : len(row)] = torch.tensor(row), 1
    values = model(ids, mask)
    return [values[i, flags] for i, (_, flags, _) in enumerate(inputs)]


# model = load(snapshot_path("Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"))
# scores, pieces = model.step_scores(tokenizer, question, answer); s, lines = steps(scores, pieces)
# first_flag(s), aggregate(s)
`;export{e as default};
