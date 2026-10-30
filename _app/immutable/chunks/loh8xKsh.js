const e=`# EAGLE on Vicuna-7B v1.3, the models in the Efficiency Explainer's EAGLE tab, written as plain PyTorch blocks.
# yuhuili/EAGLE-Vicuna-7B-v1.3 is a one-layer draft model that predicts Vicuna's next hidden state from the current
# one and the next token; run on its own predictions it grows a tree (EAGLE-2's dynamic tree) that
# lmsys/vicuna-7b-v1.3 checks in one forward with tree attention.
# Each "# %% [id]" section is one block of the diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:  # Vicuna-7B v1.3: LLaMA-7B's shape; the draft layer has the same width and heads
    vocab_size: int = 32000
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # 128 dimensions per head
    d_mlp: int = 11008
    rope_theta: float = 10000.0
    eps: float = 1e-6
    depth: int = 5  # expansion rounds after depth 1 (EAGLE's evaluation settings for Vicuna)
    topk: int = 10
    tree_tokens: int = 60  # the root and 59 drafted tokens


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


# %% [draft] Draft model
class DraftLayer(nn.Module):
    """A Llama block whose attention reads its input without an RMSNorm (the input is fc's output)."""

    def __init__(self, c: Config):
        super().__init__()
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x, positions, mask, cache=None):
        a, new = self.attention(x, positions, mask, cache)
        x = x + a
        return x + self.mlp(self.norm2(x)), new


class EagleDraft(nn.Module):
    """f_t (Vicuna's final-norm hidden state at t) and e(x_{t+1}) (the next token's embedding, Vicuna's own table)
    → fc → one layer → f̂_{t+1}, a prediction of Vicuna's next hidden state. Vicuna's LM head reads it."""

    def __init__(self, c: Config):
        super().__init__()
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)  # a frozen copy of Vicuna's
        self.fc = nn.Linear(2 * c.d_model, c.d_model)
        self.layer = DraftLayer(c)

    def forward(self, features, ids, positions, mask, cache=None):
        """features: (seq, d_model); ids: (seq,), each the token one place after its feature. mask: (seq, cached +
        seq). Returns the predicted features and the new keys and values of the draft's own cache."""
        x = self.fc(torch.cat((self.embedding(ids), features), dim=-1))
        return self.layer(x, positions, mask, cache)


# %% [tree] Dynamic draft tree
def grow_tree(draft, lm_head, cache, features, ids, c: Config):
    """EAGLE-2's tree. Catch up: Vicuna's features of the tokens kept last step go through the draft; its last
    output gives depth 1 (top-k after the root). Then \`depth\` rounds: feed the k best nodes so far, each with its
    parent's predicted feature, and give each fed node k children scored by the draft's log-probabilities summed
    along the path. Keep the tree_tokens − 1 best of all candidates. Returns their tokens and parents (0 = the
    root), and the draft cache with the caught-up positions only."""
    m, cached = len(ids), 0 if cache is None else cache[0].shape[2]
    causal = torch.cat((torch.ones(m, cached, dtype=torch.bool), torch.ones(m, m, dtype=torch.bool).tril()), dim=1)
    positions = torch.arange(cached, cached + m, device=ids.device)
    out, new = draft(features, ids, positions, causal.to(ids.device), cache)
    cache = new if cache is None else (torch.cat((cache[0], new[0]), dim=2), torch.cat((cache[1], new[1]), dim=2))
    position = cached + m  # the root's position
    top = lm_head(out[-1]).float().log_softmax(dim=-1).topk(c.topk)
    scores, all_scores, all_tokens = top.values, [top.values], [top.indices]
    parent_of_row = [-1]  # every row of k candidates shares one parent (a flat candidate index, -1 = root)
    fed_tokens, fed_features, fed_index = top.indices, out[-1].expand(c.topk, -1), torch.arange(c.topk)
    seen = torch.eye(c.topk, dtype=torch.bool)  # fed nodes × tree nodes fed so far: ancestors and self
    local = cache
    for i in range(c.depth):
        mask = torch.cat((torch.ones(c.topk, position, dtype=torch.bool), seen), dim=1).to(ids.device)
        positions = torch.full((c.topk,), position + i, device=ids.device)  # a depth shares one position
        out, new = draft(fed_features, fed_tokens, positions, mask, local)
        local = (torch.cat((local[0], new[0]), dim=2), torch.cat((local[1], new[1]), dim=2))
        parent_of_row += fed_index.tolist()
        top = lm_head(out).float().log_softmax(dim=-1).topk(c.topk, dim=-1)  # (k fed nodes, k children)
        path_scores = top.values + scores[:, None]
        best = path_scores.flatten().topk(c.topk)
        rows = best.indices // c.topk
        scores, fed_tokens, fed_features = best.values, top.indices.flatten()[best.indices], out[rows]
        fed_index = c.topk + i * c.topk * c.topk + best.indices.cpu()  # flat index of each newly fed node
        seen = torch.cat((seen[rows.cpu()], torch.eye(c.topk, dtype=torch.bool)), dim=1)
        all_scores.append(path_scores.flatten())
        all_tokens.append(top.indices.flatten())
    all_scores, all_tokens = torch.cat(all_scores), torch.cat(all_tokens)
    chosen = all_scores.topk(c.tree_tokens - 1).indices.sort().values.tolist()  # in depth order
    node_of = {flat: node + 1 for node, flat in enumerate(chosen)}
    parents = [-1] + [node_of.get(parent_of_row[flat // c.topk], 0) for flat in chosen]
    return all_tokens[chosen], parents, cache


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


# %% [model] EAGLE decoding
@torch.no_grad()
def eagle_generate(vicuna, draft, prompt, max_new_tokens, eos=2):
    """Greedy decoding with EAGLE: per step one draft catch-up, \`depth\` draft rounds and one Vicuna forward."""
    c, n = vicuna.c, len(prompt)
    hidden, logits, cache = vicuna(prompt, torch.arange(n), torch.ones(n, n, dtype=torch.bool).tril())
    root = logits[-1].argmax()
    features, ids = hidden, torch.cat((prompt[1:], root[None]))  # f_t with x_{t+1}
    draft_cache, out = None, []
    while len(out) < max_new_tokens:
        drafted, parents, draft_cache = grow_tree(draft, vicuna.lm_head, draft_cache, features, ids, c)
        tokens = torch.cat((root[None], drafted))
        depth = torch.tensor([0] * len(parents))
        for i, parent in enumerate(parents):
            depth[i] = 0 if parent < 0 else depth[parent] + 1
        mask = torch.eye(len(parents), dtype=torch.bool)  # tree attention: itself and its ancestors
        for i, parent in enumerate(parents):
            if parent >= 0:
                mask[i] |= mask[parent]
        cached = cache[0][0].shape[2]
        hidden, logits, new = vicuna(tokens, cached + depth, mask, cache)
        argmax = logits.argmax(dim=-1)
        kept = accept(tokens.tolist(), parents, argmax.tolist())
        cache = extend(cache, new, kept)
        root = argmax[kept[-1]]
        features, ids = hidden[kept], torch.cat((tokens[kept][1:], root[None]))  # Vicuna's features feed the draft
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
