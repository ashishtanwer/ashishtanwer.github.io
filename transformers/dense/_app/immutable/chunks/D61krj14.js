const e=`# R1-Distill, reasoning traces into a small model: the Post-training Explainer's R1-Distill tab as plain PyTorch.
# DeepSeek-R1 writes long <think> traces; DeepSeek-R1-Distill-Qwen-1.5B is Qwen2.5-Math-1.5B fine-tuned on 800k
# such samples with the ordinary next-token loss (SFT, no RL). Here: the prompt format, sampling a trace with
# temperature and top-p, scoring a trace token by token under any causal LM, and one SFT step on a trace.
# \`model\` maps ids (batch, seq) to logits (batch, seq, vocab), here a Qwen2 decoder (28 layers, width 1536).
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 151936
    bos: str = "<｜begin▁of▁sentence｜>"
    directive: str = "Please reason step by step, and put your final answer within \\\\boxed{}."
    temperature: float = 0.6  # DeepSeek's recommended sampling settings
    top_p: float = 0.95
    lr: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


# %% [template] The prompt
def r1_prompt(question: str, c: Config) -> str:
    """R1-Distill's chat format, with the reply forced to open a <think> block (the model card's advice):
    no system prompt, the instruction inside the user turn."""
    return f"{c.bos}<｜User｜>{question}\\n{c.directive}<｜Assistant｜><think>\\n"


# %% [sample] Sampling with temperature and top-p
def next_token_probs(logits, c: Config):
    """softmax(z / T), then keep the smallest set of most likely tokens whose probability reaches top_p
    (the nucleus) and renormalise; logits: (..., vocab)."""
    p = F.softmax(logits / c.temperature, dim=-1)
    sorted_p, order = p.sort(dim=-1, descending=True)
    before = sorted_p.cumsum(-1) - sorted_p  # probability of the tokens ranked above each one
    sorted_p = torch.where(before < c.top_p, sorted_p, torch.zeros_like(sorted_p))
    kept = torch.zeros_like(p).scatter(-1, order, sorted_p)
    return kept / kept.sum(-1, keepdim=True)


def sample_trace(model, prompt_ids, c: Config, max_new: int, end_id: int, generator=None):
    """Draw one token at a time from next_token_probs until the end token (no KV cache: plain, slow)."""
    ids = prompt_ids
    for _ in range(max_new):
        probs = next_token_probs(model(ids[None])[0, -1], c)
        token = torch.multinomial(probs, 1, generator=generator)
        ids = torch.cat([ids, token])
        if token.item() == end_id:
            break
    return ids[len(prompt_ids):]


# %% [trace-logprobs] Scoring a trace
def trace_logprobs(model, prompt_ids, trace_ids):
    """log p(y_t | x, y_<t) for every trace token, at temperature 1 (the distribution SFT trains): one forward
    pass over prompt + trace; position j − 1 predicts token j."""
    ids = torch.cat([prompt_ids, trace_ids])[None]
    log_p = F.log_softmax(model(ids)[0, len(prompt_ids) - 1 : -1].float(), dim=-1)  # (trace, vocab)
    return log_p.gather(-1, trace_ids[:, None])[:, 0]


def gain(student_logp, base_logp):
    """How much likelier distillation made each token, in nats: log p_student − log p_base."""
    return student_logp - base_logp


# %% [sft-on-traces] SFT on a teacher's trace
def sft_loss(model, prompt_ids, trace_ids):
    """The distillation objective: the mean −log p of the teacher's trace, with the prompt not scored."""
    return -trace_logprobs(model, prompt_ids, trace_ids).mean()


class AdamW:
    def __init__(self, params, c: Config):
        self.params, self.c, self.t = list(params), c, 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        (b1, b2), lr = self.c.betas, self.c.lr
        for p, m, v in zip(self.params, self.m, self.v):
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
            p.sub_(lr * (m / (1 - b1**self.t)) / ((v / (1 - b2**self.t)).sqrt() + self.c.eps))


def sft_step(model, optimizer, prompt_ids, trace_ids):
    for p in optimizer.params:
        p.grad = None
    loss = sft_loss(model, prompt_ids, trace_ids)
    loss.backward()
    optimizer.step()
    return loss.detach()
`;export{e as default};
