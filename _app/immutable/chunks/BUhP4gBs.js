const e=`# Medusa on Vicuna-7B v1.3, the models in the Efficiency Explainer's Medusa tab, written as plain PyTorch blocks.
# FasterDecoding/medusa-vicuna-7b-v1.3 adds five heads to lmsys/vicuna-7b-v1.3; a fixed tree of their top-10
# guesses is checked in one Vicuna forward with tree attention, and the longest path Vicuna agrees with is kept.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Vicuna-7B v1.3: LLaMA-7B's shape
    vocab_size: int = 32000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # 128 dimensions per head
    d_mlp: int = 11008
    rope_theta: float = 10000.0
    eps: float = 1e-6
    medusa_heads: int = 5  # the released checkpoint has five (its config.json says two)
    topk: int = 10  # candidates taken from each head


# Medusa's tree for a 7B Vicuna (medusa_choices.vicuna_7b_stage2): each entry is a path of ranks, (0, 1) = head 1's
# top token, then head 2's second token. 63 nodes below the root, at most 4 deep.
VICUNA_7B_STAGE2 = [(0,), (0, 0), (1,), (0, 1), (0, 0, 0), (1, 0), (2,), (0, 2), (0, 0, 1), (0, 3), (3,), (0, 1, 0),
                    (2, 0), (4,), (0, 0, 2), (0, 4), (1, 1), (1, 0, 0), (0, 0, 0, 0), (5,), (0, 0, 3), (0, 5),
                    (0, 2, 0), (3, 0), (0, 1, 1), (0, 6), (6,), (0, 7), (0, 0, 4), (4, 0), (1, 2), (0, 8), (7,),
                    (0, 3, 0), (0, 0, 0, 1), (0, 0, 5), (2, 1), (0, 0, 6), (1, 0, 1), (0, 0, 1, 0), (2, 0, 0), (5, 0),
                    (0, 9), (0, 1, 2), (8,), (0, 4, 0), (0, 2, 1), (1, 3), (0, 0, 7), (0, 0, 0, 2), (0, 0, 8),
                    (1, 1, 0), (0, 1, 0, 0), (6, 0), (9,), (0, 1, 3), (0, 0, 0, 3), (1, 0, 2), (0, 5, 0), (3, 1),
                    (0, 0, 2, 0), (7, 0), (1, 4)]


# %% [block] Vicuna block
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):  # in float32, as Llama does, then back to x's dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


def rope(x, positions, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d). positions: (seq,) — tree
    tokens share positions: a node sits at prefix length + its depth."""
    d = x.shape[-1]
    frequencies = (1.0 / theta ** (torch.arange(0, d, 2).float() / d)).to(x.device)  # on the CPU, as Llama does
    angles = positions.float()[:, None] * frequencies  # (seq, d/2)
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class Attention(nn.Module):
    """Multi-head attention over the cached keys and values plus the new tokens, under any boolean mask."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, positions, mask, cache=None):
        """x: (seq, d_model). mask: (seq, cached + seq), True = may attend. Returns the output and the new (k, v)."""
        c, t = self.c, x.shape[0]
        q = self.q(x).view(1, t, c.n_heads, -1).transpose(1, 2)  # (1, heads, seq, head_dim), as Llama keeps it
        k = self.k(x).view(1, t, c.n_heads, -1).transpose(1, 2)
        v = self.v(x).view(1, t, c.n_heads, -1).transpose(1, 2)
        q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
        keys = k if cache is None else torch.cat((cache[0], k), dim=2)
        values = v if cache is None else torch.cat((cache[1], v), dim=2)
        o = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(t, -1)), (k, v)


class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x)), 4096 → 11008 → 4096."""

    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm residual block: x + Attention(RMSNorm(x)), then x + MLP(RMSNorm(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, positions, mask, cache=None):
        a, new = self.attention(self.norm1(x), positions, mask, cache)
        x = x + a
        return x + self.mlp(self.norm2(x)), new


# %% [heads] Medusa heads
class MedusaHead(nn.Module):
    """One residual block, x + SiLU(W x + b), then an LM head of its own. Head i is trained to guess the token
    i + 1 places past the one Vicuna's LM head predicts, from the same hidden state."""

    def __init__(self, c: Config):
        super().__init__()
        self.linear = nn.Linear(c.d_model, c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, h):  # h: Vicuna's last hidden state after the final RMSNorm, (..., d_model)
        return self.lm_head(h + F.silu(self.linear(h)))  # logits for token t + 1 + i


class MedusaHeads(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = nn.ModuleList(MedusaHead(c) for _ in range(c.medusa_heads))

    def forward(self, h):  # (d_model,) → (heads, vocab); no head sees another head's guess
        return torch.stack([head(h) for head in self.heads])


# %% [tree] Candidate tree
def build_tree(choices, topk):
    """Medusa's buffers. Node 0 is the root (Vicuna's own next token); the others are the choices sorted by depth,
    then by ranks. slot: where each node's token sits in [root, head 1's top-k, head 2's top-k, …]."""
    ordered = sorted(choices, key=lambda choice: (len(choice), choice))
    parents = [-1] + [ordered.index(choice[:-1]) + 1 if len(choice) > 1 else 0 for choice in ordered]
    depth = torch.tensor([0] + [len(choice) for choice in ordered])
    slot = torch.tensor([0] + [1 + (len(choice) - 1) * topk + choice[-1] for choice in ordered])
    mask = torch.eye(len(parents), dtype=torch.bool)  # a node sees itself and its ancestors
    for i, parent in enumerate(parents):
        if parent >= 0:
            mask[i] |= mask[parent]
    return parents, depth, slot, mask


def candidates(root, head_logits, slot, topk):
    """The tree's tokens: the root, then each node's rank from its depth's head."""
    top = head_logits.topk(topk, dim=-1).indices  # (heads, topk)
    return torch.cat((root[None], top.flatten()))[slot]


# %% [tree-attention] Tree attention
class Vicuna(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, ids, positions, mask, cache=None):
        """ids, positions: (seq,). mask: (seq, seq) over the new tokens; every new token also sees the whole cache.
        A causal mask makes this a prefill or a greedy step; a tree mask scores every path of a candidate tree at
        once. Returns the final-norm hidden states (what the heads read), the logits and the new keys and values."""
        cached = 0 if cache is None else cache[0][0].shape[2]
        full = torch.cat((torch.ones(len(ids), cached, dtype=torch.bool, device=ids.device), mask), dim=1)
        x, new = self.embedding(ids), []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, positions, full, None if cache is None else cache[i])
            new.append(kv)
        h = self.norm(x)
        return h, self.lm_head(h), new


def extend(cache, new, keep=None):
    """Append the new tokens' keys and values to the cache: all of them, or only the tree nodes in \`keep\`."""
    if keep is not None:
        new = [(k[:, :, keep], v[:, :, keep]) for k, v in new]
    if cache is None:
        return new
    return [(torch.cat((ck, k), dim=2), torch.cat((cv, v), dim=2)) for (ck, cv), (k, v) in zip(cache, new)]


# %% [verify] Greedy acceptance
def accept(tokens, parents, argmax):
    """Along each root-to-leaf path, keep tokens while each equals Vicuna's argmax at its parent; the longest kept
    path wins (the first leaf on ties). Returns the kept nodes, root first. Every kept token is the one plain
    greedy decoding would have produced, so the answer does not change."""
    is_parent = set(parents)
    best = [0]
    for leaf in range(len(parents)):
        if leaf in is_parent:
            continue
        path = [leaf]
        while parents[path[0]] >= 0:
            path.insert(0, parents[path[0]])
        kept = 1
        while kept < len(path) and tokens[path[kept]] == argmax[path[kept - 1]]:
            kept += 1
        if kept > len(best):
            best = path[:kept]
    return best


# %% [model] Medusa decoding
@torch.no_grad()
def medusa_generate(vicuna, heads, prompt, max_new_tokens, choices=VICUNA_7B_STAGE2, eos=2):
    """Greedy decoding with Medusa: one Vicuna forward per step, each adding the root and the kept drafts."""
    topk = vicuna.c.topk
    parents, depth, slot, mask = build_tree(choices, topk)
    n = len(prompt)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    hidden, logits, cache = vicuna(prompt, torch.arange(n), causal)
    h, last, out = hidden[-1], logits[-1], []
    while len(out) < max_new_tokens:
        tokens = candidates(last.argmax(), heads(h), slot, topk)
        cached = cache[0][0].shape[2]
        hidden, logits, new = vicuna(tokens, cached + depth, mask, cache)
        kept = accept(tokens.tolist(), parents, logits.argmax(dim=-1).tolist())
        cache = extend(cache, new, kept)
        h, last = hidden[kept[-1]], logits[kept[-1]]  # the next root and the next guesses come from here
        out += tokens[kept].tolist()
        if eos in out:
            return out[: out.index(eos) + 1]
    return out


@torch.no_grad()
def greedy_generate(vicuna, prompt, max_new_tokens, eos=2):
    """Plain greedy decoding, one Vicuna forward per token: what Medusa must reproduce."""
    n = len(prompt)
    _, logits, cache = vicuna(prompt, torch.arange(n), torch.ones(n, n, dtype=torch.bool).tril())
    out = []
    while len(out) < max_new_tokens:
        out.append(int(logits[-1].argmax()))
        if out[-1] == eos:
            break
        position = torch.tensor([n + len(out) - 1])
        _, logits, new = vicuna(torch.tensor(out[-1:]), position, torch.ones(1, 1, dtype=torch.bool), cache)
        cache = extend(cache, new)
    return out
`;export{e as default};
