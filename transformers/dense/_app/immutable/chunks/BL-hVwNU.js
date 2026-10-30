const e=`# DP attention for DeepSeek-V2-Lite's MLA, the model in the DP-attention tab, as plain PyTorch blocks.
# Four ranks serve one MLA + MoE layer two ways: tensor parallel (heads split, every rank keeps the whole
# batch's latent cache) and DP attention (requests split, each rank keeps only its own). The ranks are
# lists of tensors in one process; the collectives are written out. Each "# %% [id]" section is one block.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # DeepSeek-V2-Lite
    d_model: int = 2048
    n_heads: int = 16
    latent: int = 512  # kv_lora_rank: the cached c_KV per token
    rope: int = 64  # the rotary key, also cached, shared by every head
    content: int = 128  # per-head content (no-RoPE) query and key
    value: int = 128
    n_experts: int = 64
    n_shared: int = 2
    top_k: int = 6
    d_expert: int = 1408
    eps: float = 1e-6
    rope_theta: float = 10000.0
    yarn_factor: float = 40.0
    yarn_mscale: float = 0.707
    ranks: int = 4
    bytes_per_element: int = 2  # bfloat16

    @property
    def softmax_scale(self):
        """DeepSeek's: 1/√(content + rope) times YaRN's mscale squared."""
        m = 0.1 * self.yarn_mscale * math.log(self.yarn_factor) + 1.0
        return (self.content + self.rope) ** -0.5 * m * m


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


def rotate(x, freqs):
    """RoPE on interleaved pairs as complex numbers; freqs: (seq, rope/2) unit complex (YaRN)."""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(pairs * freqs).flatten(-2)


# %% [mla] MLA: one latent per token
class MLA(nn.Module):
    """Multi-head latent attention. The cache holds, per token, a 512-D latent c_KV and a 64-D rotary
    key; each head expands the latent into its own key and value (W↑K, W↑V). \`heads\` limits the
    computation to some heads, as one tensor-parallel rank does."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.n_heads * (c.content + c.rope), bias=False)
        self.kv_down = nn.Linear(c.d_model, c.latent + c.rope, bias=False)
        self.kv_norm = RMSNorm(c.latent, c.eps)
        self.kv_up = nn.Linear(c.latent, c.n_heads * (c.content + c.value), bias=False)
        self.out = nn.Linear(c.n_heads * c.value, c.d_model, bias=False)

    def cache(self, x, freqs):
        """What MLA caches: (seq, 512) latents and (seq, 64) rotated keys, the same for every head."""
        latent, k_rope = self.kv_down(x).split([self.c.latent, self.c.rope], dim=-1)
        return self.kv_norm(latent), rotate(k_rope, freqs)

    def forward(self, x, freqs, heads=None):  # x: (seq, d_model) normed; returns o_proj's (partial) output
        c, t = self.c, x.shape[0]
        heads = torch.arange(c.n_heads) if heads is None else torch.as_tensor(heads)
        h = len(heads)
        wq = self.q.weight.view(c.n_heads, c.content + c.rope, -1)[heads].flatten(0, 1)
        q = F.linear(x, wq).view(t, h, -1).transpose(0, 1)  # (heads, seq, 192)
        q_content, q_rope = q.split([c.content, c.rope], dim=-1)
        latent, k_rope = self.cache(x, freqs)
        wkv = self.kv_up.weight.view(c.n_heads, c.content + c.value, -1)[heads].flatten(0, 1)
        k_content, v = F.linear(latent, wkv).view(t, h, -1).transpose(0, 1).split([c.content, c.value], dim=-1)
        query = torch.cat((q_content, rotate(q_rope, freqs)), dim=-1)
        key = torch.cat((k_content, k_rope.expand(h, -1, -1)), dim=-1)
        scores = query @ key.transpose(-1, -2) * c.softmax_scale
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        wo = self.out.weight.view(c.d_model, c.n_heads, c.value)[:, heads].flatten(1)
        return F.linear((weights @ v).transpose(0, 1).reshape(t, -1), wo), weights


# %% [collectives] Collectives, written out
def all_reduce(parts):
    """Every rank ends with the sum of every rank's tensor. Ring: each rank sends 2(N−1)/N of it."""
    total = torch.stack(parts).sum(dim=0)
    return [total.clone() for _ in parts]


def all_gather(parts):
    """Every rank ends with every rank's rows, in rank order."""
    rows = torch.cat(parts)
    return [rows.clone() for _ in parts]


def reduce_scatter(parts, sizes):
    """Sum every rank's tensor, then give rank r the rows it owns (sizes[r] of them, in rank order)."""
    return list(torch.stack(parts).sum(dim=0).split(sizes))


# %% [tensor-parallel] Tensor parallel: heads split
def tensor_parallel_attention(mla, requests, freqs, ranks=4):
    """Rank r computes heads 4r … 4r+3 of every request. Every rank needs the latent of every token,
    so every rank computes and stores the whole batch's cache. Returns each request's output and the
    latent rows each rank holds."""
    c = mla.c
    per = c.n_heads // ranks
    outputs, held = [], [0] * ranks
    for x, f in zip(requests, freqs):
        parts = [mla(x, f, range(r * per, (r + 1) * per))[0] for r in range(ranks)]  # o_proj slices
        outputs.append(all_reduce(parts)[0])
        for r in range(ranks):
            held[r] += x.shape[0]  # the whole latent cache of this request
    return outputs, held


# %% [dp-attention] DP attention: requests split
def dp_attention(mla, requests, freqs, ranks=4):
    """Request i goes to rank i mod 4, which runs every head for it with the full attention weights and
    keeps only its own requests' latents. Nothing is exchanged inside attention."""
    outputs, held = [None] * len(requests), [0] * ranks
    for i, (x, f) in enumerate(zip(requests, freqs)):
        rank = i % ranks
        outputs[i] = mla(x, f)[0]
        held[rank] += x.shape[0]
    return outputs, held


# %% [moe] MoE, expert parallel
class Expert(nn.Module):
    def __init__(self, d_model, width):
        super().__init__()
        self.gate = nn.Linear(d_model, width, bias=False)
        self.up = nn.Linear(d_model, width, bias=False)
        self.down = nn.Linear(width, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """64 routed experts (top 6 of a softmax, not renormalized) plus 2 shared experts as one MLP."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.router = nn.Linear(c.d_model, c.n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(c.d_model, c.d_expert) for _ in range(c.n_experts))
        self.shared = Expert(c.d_model, c.d_expert * c.n_shared)

    def route(self, y):
        return self.router(y).softmax(dim=-1).topk(self.c.top_k, dim=-1, sorted=False)

    def part(self, y, rank, ranks=4):
        """Rank \`rank\`'s share for tokens y: its experts (16 of 64) and its quarter of the shared
        experts' columns. The ranks' shares add up to the MoE."""
        weights, ids = self.route(y)
        per = self.c.n_experts // ranks
        out = torch.zeros_like(y)
        for e in range(rank * per, (rank + 1) * per):
            token, slot = (ids == e).nonzero(as_tuple=True)
            if len(token):
                out.index_add_(0, token, self.experts[e](y[token]) * weights[token, slot, None])
        width = self.shared.gate.weight.shape[0] // ranks
        cols = slice(rank * width, (rank + 1) * width)
        hidden = F.silu(F.linear(y, self.shared.gate.weight[cols])) * F.linear(y, self.shared.up.weight[cols])
        return out + F.linear(hidden, self.shared.down.weight[:, cols])

    def forward(self, y):
        return sum(self.part(y, r) for r in range(4))


# %% [layer] One layer served both ways
def serve(layer, requests, freqs, mode, ranks=4):
    """requests: each request's hidden states (seq, d_model). Returns each request's layer output.
    TP: all-reduce after attention, every rank runs the MoE's share on every token, all-reduce.
    DP: attention on each request's own rank, all-gather, the MoE's shares, reduce-scatter."""
    xs = [layer.norm1(h) for h in requests]
    if mode == "tp":
        attention, _ = tensor_parallel_attention(layer.mla, xs, freqs, ranks)
    else:
        attention, _ = dp_attention(layer.mla, xs, freqs, ranks)
    hs = [h + a for h, a in zip(requests, attention)]
    # DP keeps each rank's rows together; TP has every row everywhere, in request order
    order = [i for r in range(ranks) for i in range(len(hs)) if i % ranks == r] if mode == "dp" else list(range(len(hs)))
    if mode == "dp":  # rank r holds its requests' rows; the all-gather gives every rank every row
        local = [torch.cat([layer.norm2(hs[i]) for i in order if i % ranks == r]) for r in range(ranks)]
        y = all_gather(local)[0]
        sizes = [len(t) for t in local]
        moe = torch.cat(reduce_scatter([layer.moe.part(y, r, ranks) for r in range(ranks)], sizes))
    else:
        y = torch.cat([layer.norm2(h) for h in hs])
        moe = all_reduce([layer.moe.part(y, r, ranks) for r in range(ranks)])[0]
    starts = torch.tensor([0] + [len(hs[i]) for i in order]).cumsum(0)
    out = [None] * len(hs)
    for k, i in enumerate(order):
        out[i] = hs[i] + moe[starts[k] : starts[k + 1]]
    return out


class Layer(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.mla = MLA(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.moe = MoE(c)


# %% [memory] Memory per rank
def memory_per_rank(c: Config, tokens_per_rank, mode):
    """Bytes each rank holds for one layer (bfloat16): its latent cache rows (576 per token) and its
    weights. TP splits the per-head projections and replicates the latent cache; DP replicates the
    attention weights and splits the cache. The MoE is expert parallel either way."""
    per_token = (c.latent + c.rope) * c.bytes_per_element
    heads = c.n_heads * (c.content + c.rope) * c.d_model + c.n_heads * (c.content + c.value) * c.latent
    heads += c.d_model * c.n_heads * c.value  # q, kv_up and out: split by heads under TP
    shared = (c.latent + c.rope) * c.d_model + c.latent  # kv_down and its norm: every rank
    attention = (heads // c.ranks if mode == "tp" else heads) + shared
    expert = 3 * c.d_model * c.d_expert
    moe = c.n_experts // c.ranks * expert + 3 * c.d_model * c.d_expert * c.n_shared // c.ranks + c.n_experts * c.d_model
    return {
        "kv": [n * per_token for n in tokens_per_rank],
        "attention": attention * c.bytes_per_element,
        "moe": moe * c.bytes_per_element,
    }


# out_tp = serve(Layer(), hidden_states, freqs, "tp"); out_dp = serve(Layer(), hidden_states, freqs, "dp")
`;export{e as default};
