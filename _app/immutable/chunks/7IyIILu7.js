const e=`# Expert parallelism for OLMoE-1B-7B (0125)'s first MoE layer, the model in the EP tab, as plain PyTorch blocks.
# The 64 experts are spread over the ranks, 16 each; every rank keeps the router and its own tokens. A token's
# copies travel to the ranks that hold its eight chosen experts in one all-to-all (dispatch), are computed
# there, and come back in a second all-to-all (combine), as Megatron-LM's all-to-all token dispatcher does.
# Every rank is its own process; \`comm\` is torch.distributed (the recorder passes one that also logs each
# call). Each "# %% [id]" section is one block.
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # OLMoE-1B-7B-0125
    d_model: int = 2048
    d_expert: int = 1024
    n_experts: int = 64
    top_k: int = 8
    norm_topk_prob: bool = False  # OLMoE keeps the softmax probabilities of its top 8 as they are
    ranks: int = 4

    @property
    def per_rank(self):
        return self.n_experts // self.ranks


class Comm:
    """What a rank calls to talk to the others."""

    def all_to_all(self, x, send_rows, recv_rows, label=""):
        """Rows of x go out in order, send_rows[j] of them to rank j; recv_rows[i] rows come in from rank i."""
        out = x.new_empty((sum(recv_rows), *x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), recv_rows, send_rows)
        return out


# %% [router] The router: a softmax over 64 experts, the top 8
def route(c: Config, x, gate):
    """x: (n, 2048) → each token's eight experts and their weights. The router is one linear layer kept
    on every rank, so a rank routes its own tokens without asking anyone."""
    probs = F.softmax(x @ gate.T, dim=-1, dtype=torch.float)  # (n, 64)
    weights, experts = probs.topk(c.top_k, dim=-1)  # (n, 8) each
    if c.norm_topk_prob:
        weights = weights / weights.sum(-1, keepdim=True)
    return experts, weights, probs


# %% [placement] 16 experts on each rank
def owner(c: Config, expert):
    """Expert e lives on rank e // 16: rank 0 holds experts 0–15, rank 1 16–31, …"""
    return expert // c.per_rank


class Expert(nn.Module):
    """One OLMoE expert, a SwiGLU MLP: down(silu(gate(x)) · up(x)), 2048 → 1024 → 2048."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate_proj = nn.Linear(c.d_model, c.d_expert, bias=False)
        self.up_proj = nn.Linear(c.d_model, c.d_expert, bias=False)
        self.down_proj = nn.Linear(c.d_expert, c.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# %% [dispatch] All-to-all dispatch
def dispatch(c: Config, x, experts, comm):
    """One copy of a token per chosen expert, sorted by expert, so the copies for each rank are
    consecutive. A first all-to-all tells every rank how many rows of each of its experts are coming
    (16 counts from each rank); the second carries the rows."""
    flat = experts.flatten()  # (n·8,) expert of each copy
    order = flat.argsort(stable=True)  # copies sorted by expert, hence by rank
    rows = x[order // c.top_k]  # (n·8, 2048)
    per_expert = torch.bincount(flat, minlength=c.n_experts)  # copies per expert, from this rank
    counts = comm.all_to_all(per_expert.view(c.ranks, c.per_rank), [1] * c.ranks, [1] * c.ranks, "counts")
    send = per_expert.view(c.ranks, c.per_rank).sum(1).tolist()  # rows to each rank
    recv = counts.sum(1).tolist()  # rows from each rank
    received = comm.all_to_all(rows, send, recv, "dispatch")  # by source rank, by expert inside
    return received, counts, order, send, recv


# %% [experts] This rank's experts on the rows it received
def run_experts(c: Config, received, counts, experts: nn.ModuleList):
    """counts[i, e]: rows rank i sent for local expert e, in that order. Each local expert runs once on
    all of its rows, from every rank."""
    local = torch.arange(c.per_rank).repeat(c.ranks)  # the expert of each (source, expert) block
    expert_of_row = torch.repeat_interleave(local, counts.flatten())
    out = torch.empty_like(received)
    for e, expert in enumerate(experts):
        mine = expert_of_row == e
        out[mine] = expert(received[mine])
    return out


# %% [combine] All-to-all combine
def combine(c: Config, out, order, weights, send, recv, n, comm):
    """The second all-to-all is the first one backwards: every row returns to the rank it came from, in
    the order it left. Each token's eight outputs are then weighted by the router and added."""
    back = comm.all_to_all(out, recv, send, "combine")  # (n·8, 2048), sorted by expert as sent
    w = weights.flatten()[order].to(back.dtype)
    y = torch.zeros(n, c.d_model, dtype=back.dtype)
    return y.index_add_(0, order // c.top_k, back * w[:, None])


# %% [model] The MoE layer on one rank
class ExpertParallelMoE(nn.Module):
    """OLMoE's sparse MoE block as one of the ranks runs it: the whole router, its 16 experts."""

    def __init__(self, c: Config, rank: int, comm):
        super().__init__()
        self.c, self.rank, self.comm = c, rank, comm
        self.gate = nn.Linear(c.d_model, c.n_experts, bias=False)
        self.first = rank * c.per_rank
        self.experts = nn.ModuleList(Expert(c) for _ in range(c.per_rank))

    def forward(self, x):  # x: (n, 2048), this rank's tokens after attention and the norm
        experts, weights, probs = route(self.c, x, self.gate.weight)
        received, counts, order, send, recv = dispatch(self.c, x, experts, self.comm)
        out = run_experts(self.c, received, counts, self.experts)
        y = combine(self.c, out, order, weights, send, recv, len(x), self.comm)
        self.routed = dict(experts=experts, weights=weights, probs=probs, counts=counts, send=send, recv=recv)
        return y

    def load(self, read):
        """read(name) returns that checkpoint tensor of layer 0's mlp (names as in the checkpoint)."""
        state = {"gate.weight": read("gate.weight")}
        for i in range(self.c.per_rank):
            for p in ("gate_proj", "up_proj", "down_proj"):
                state[f"experts.{i}.{p}.weight"] = read(f"experts.{self.first + i}.{p}.weight")
        self.load_state_dict(state, strict=True)
        return self
`;export{e as default};
