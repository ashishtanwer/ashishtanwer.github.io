const e=`# Pipeline parallelism for GPT-2 (124M), the model in the PP tab, as plain PyTorch blocks.
# The twelve blocks are cut into consecutive stages, one per rank; a batch is cut into micro-batches that
# follow each other through the stages. GPipe runs every micro-batch forward, then every one backward;
# 1F1B (PipeDream-Flush, Megatron-LM's default schedule) starts each micro-batch's backward as soon as it can,
# so a stage holds at most (stages - its index) micro-batches' activations. Every rank is its own process;
# \`comm\` sends and receives between neighbouring stages. Each "# %% [id]" section is one block.
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # GPT-2 (124M)
    d_model: int = 768
    n_heads: int = 12
    d_mlp: int = 3072
    n_layers: int = 12
    vocab: int = 50257
    context: int = 1024
    eps: float = 1e-5
    stages: int = 4
    micro_batches: int = 8
    micro_tokens: int = 128  # one sequence of 128 tokens per micro-batch


class Comm:
    """What a rank calls to talk to the others: posted sends and receives, waited for together."""

    def exchange(self, sends, recvs):  # sends: [(tensor, dst, label)], recvs: [(shape, src, label)]
        out = [torch.empty(shape) for shape, _, _ in recvs]
        ops = [dist.P2POp(dist.isend, t.contiguous(), dst) for t, dst, _ in sends]
        ops += [dist.P2POp(dist.irecv, buf, src) for buf, (_, src, _) in zip(out, recvs)]
        for work in dist.batch_isend_irecv(ops) if ops else []:
            work.wait()
        return out

    def all_reduce(self, x, label="", group=None):
        dist.all_reduce(x, group=group)
        return x


class Block(nn.Module):
    """One GPT-2 block, whole (the Dense tab's): causal attention and the GELU MLP, pre-LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        self.n, self.d = c.n_heads, c.d_model // c.n_heads
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attn = nn.ModuleDict(dict(c_attn=Conv1D(c.d_model, 3 * c.d_model), c_proj=Conv1D(c.d_model, c.d_model)))
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = nn.ModuleDict(dict(c_fc=Conv1D(c.d_model, c.d_mlp), c_proj=Conv1D(c.d_mlp, c.d_model)))

    def forward(self, x):  # (seq, 768)
        t = x.shape[0]
        q, k, v = self.attn.c_attn(self.ln_1(x)).view(t, 3, self.n, self.d).permute(1, 2, 0, 3)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (heads, seq, 64)
        x = x + self.attn.c_proj(y.transpose(0, 1).reshape(t, -1))
        return x + self.mlp.c_proj(F.gelu(self.mlp.c_fc(self.ln_2(x)), approximate="tanh"))


class Conv1D(nn.Module):
    """GPT-2's linear layer: x @ W + b with W stored as (in, out)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.empty(d_out))

    def forward(self, x):
        return x @ self.weight + self.bias


# %% [stages] Twelve blocks in consecutive stages
def stage_blocks(c: Config, stage: int) -> range:
    """The blocks a stage runs: an equal share of the twelve, in order."""
    per = c.n_layers // c.stages
    return range(stage * per, (stage + 1) * per)


class Stage(nn.Module):
    """One rank's part of the model. The first stage also embeds the tokens; the last applies the final
    LayerNorm and the LM head, whose matrix is the token embedding again (GPT-2 ties them), so the first
    and last stages each keep a copy of it."""

    def __init__(self, c: Config, stage: int):
        super().__init__()
        self.c, self.stage = c, stage
        self.first, self.last = stage == 0, stage == c.stages - 1
        self.layers = list(stage_blocks(c, stage))
        self.h = nn.ModuleDict({str(i): Block(c) for i in self.layers})
        if self.first or self.last:
            self.wte = nn.Embedding(c.vocab, c.d_model)
        if self.first:
            self.wpe = nn.Embedding(c.context, c.d_model)
        if self.last:
            self.ln_f = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, x, ids):
        """x: the previous stage's output (seq, 768), or None on the first stage. The last stage returns
        the micro-batch's loss, the others the hidden states they pass on."""
        if self.first:
            x = self.wte(ids) + self.wpe(torch.arange(len(ids)))
        for i in self.layers:
            x = self.h[str(i)](x)
        if not self.last:
            return x
        logits = self.ln_f(x) @ self.wte.weight.T
        return F.cross_entropy(logits[:-1], ids[1:]) / self.c.micro_batches


# %% [p2p] Sends and receives between neighbouring stages
class Pipe:
    """Megatron's p2p_communication: activations go to the next stage, their gradients come back from it.
    The paired calls post a send and a receive together, so two neighbours swapping tensors cannot
    deadlock."""

    def __init__(self, c: Config, stage: int, comm):
        self.c, self.stage, self.comm = c, stage, comm
        self.shape = (c.micro_tokens, c.d_model)
        self.prev = stage - 1 if stage > 0 else None
        self.next = stage + 1 if stage < c.stages - 1 else None

    def _go(self, send=None, send_to=None, recv_from=None, label=""):
        sends = [(send, send_to, label)] if send is not None and send_to is not None else []
        recvs = [(self.shape, recv_from, label)] if recv_from is not None else []
        if not sends and not recvs:
            return None
        got = self.comm.exchange(sends, recvs)
        return got[0] if recvs else None

    def recv_forward(self, i):
        return self._go(recv_from=self.prev, label=f"F{i}")

    def send_forward(self, y, i):
        self._go(y, self.next, label=f"F{i}")

    def recv_backward(self, i):
        return self._go(recv_from=self.next, label=f"B{i}")

    def send_backward(self, g, i):
        self._go(g, self.prev, label=f"B{i}")

    def send_forward_recv_backward(self, y, i, j):  # y of micro-batch i out, gradient of j in
        sends = [(y, self.next, f"F{i}")] if self.next is not None else []
        recvs = [(self.shape, self.next, f"B{j}")] if self.next is not None else []
        return self.comm.exchange(sends, recvs)[0] if sends else None

    def send_backward_recv_forward(self, g, i, j):  # gradient of i back, input of j in
        sends = [(g, self.prev, f"B{i}")] if self.prev is not None else []
        recvs = [(self.shape, self.prev, f"F{j}")] if self.prev is not None else []
        return self.comm.exchange(sends, recvs)[0] if sends else None


class Runner:
    """One micro-batch forward or backward on this stage. Forward keeps what backward needs (the input,
    the output and autograd's saved tensors) until that micro-batch's backward frees it."""

    def __init__(self, stage: Stage, batch):
        self.stage, self.batch = stage, batch
        self.held, self.losses = {}, {}

    def forward(self, i, x):
        if x is not None:
            x = x.detach().requires_grad_()
        y = self.stage(x, self.batch[i])
        self.held[i] = (x, y)
        if self.stage.last:
            self.losses[i] = y.detach()
            return None
        return y.detach()

    def backward(self, i, grad):
        x, y = self.held.pop(i)
        torch.autograd.backward(y, grad_tensors=None if self.stage.last else grad)
        return x.grad if x is not None else None


# %% [gpipe] GPipe: all forwards, then all backwards
def gpipe(runner: Runner, pipe: Pipe, m: int):
    """Every micro-batch forward in order, then every one backward, last first as in the GPipe paper's
    figure. Each stage holds all m micro-batches' activations at the turn."""
    for i in range(m):
        y = runner.forward(i, pipe.recv_forward(i))
        pipe.send_forward(y, i)
    for i in reversed(range(m)):
        g = runner.backward(i, pipe.recv_backward(i))
        pipe.send_backward(g, i)


# %% [1f1b] 1F1B: one forward, one backward
def one_f_one_b(runner: Runner, pipe: Pipe, m: int):
    """Megatron-LM's forward_backward_pipelining_without_interleaving. Stage s first runs (stages - s - 1)
    warm-up forwards, then alternates one forward and one backward, then drains the backwards left. The
    last stage starts a micro-batch's backward right after its forward."""
    warmup = min(pipe.c.stages - pipe.stage - 1, m)
    for i in range(warmup):
        y = runner.forward(i, pipe.recv_forward(i))
        pipe.send_forward(y, i)
    x = pipe.recv_forward(warmup) if warmup < m else None
    for k in range(m - warmup):
        i, j = warmup + k, k  # forward micro-batch i, backward micro-batch j
        y = runner.forward(i, x)
        grad = pipe.send_forward_recv_backward(y, i, j)
        g = runner.backward(j, grad)
        if k == m - warmup - 1:
            pipe.send_backward(g, j)
        else:
            x = pipe.send_backward_recv_forward(g, j, i + 1)
    for j in range(m - warmup, m):
        g = runner.backward(j, pipe.recv_backward(j))
        pipe.send_backward(g, j)


# %% [tied-embedding] The tied embedding's two copies
def sync_embedding(stage: Stage, comm, group):
    """The first stage's copy of the token embedding got the input side's gradient, the last stage's the
    LM head's; GPT-2 has one matrix, so the two gradients are summed on both (Megatron's embedding group)."""
    if stage.first or stage.last:
        comm.all_reduce(stage.wte.weight.grad, "embedding", group=group)


# %% [model] One training step on one stage
def train_step(stage: Stage, comm, batch, schedule, group, runner=Runner):
    """batch: (micro-batches, 128) token ids, which every stage reads (the last for the labels). Returns
    the micro-batches' losses on the last stage; afterwards every parameter's .grad equals the unsplit
    model's gradient for the mean loss."""
    stage.zero_grad(set_to_none=True)
    run, pipe = runner(stage, batch), Pipe(stage.c, stage.stage, comm)
    (gpipe if schedule == "gpipe" else one_f_one_b)(run, pipe, len(batch))
    sync_embedding(stage, comm, group)
    return [run.losses[i] for i in sorted(run.losses)]
`;export{e as default};
