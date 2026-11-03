const e=`# The four drafters of the Efficiency Explainer's Draft heads family side by side, written as plain PyTorch blocks:
# Medusa's heads and EAGLE's draft layer read Vicuna-7B v1.3's last hidden state; DFlash's and DSpark's block
# drafters read the outputs of five Qwen3-4B blocks. Each makes drafts its own way; all four are checked by the
# same greedy rule, so the answer is the target's greedy answer and only the acceptance length differs.
# Each "# %% [id]" section is one block of the page; the drafters' tabs show each one in full.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Vicuna:  # Vicuna-7B v1.3 (LLaMA-7B's shape), the target of Medusa and EAGLE
    vocab_size: int = 32000
    d_model: int = 4096
    n_heads: int = 32
    d_mlp: int = 11008
    rope_theta: float = 10000.0
    eps: float = 1e-6
    medusa_heads: int = 5


@dataclass
class Qwen3:  # Qwen3-4B, the target of DFlash and DSpark; their draft layers have its width, heads and MLP
    vocab_size: int = 151936
    d_model: int = 2560
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_mlp: int = 9728
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    draft_layers: int = 5
    feature_layers: tuple = (1, 9, 17, 25, 33)  # Qwen3 blocks (0-based) whose outputs both drafters read
    mask: int = 151669  # an embedding row Qwen3's tokenizer never emits: "not drafted yet"
    markov_rank: int = 256  # DSpark only


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):  # in float32, then back to x's dtype, as Llama and Qwen3 do
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(x.dtype)


class MLP(nn.Module):
    """SwiGLU: down(SiLU(gate(x)) ⊙ up(x))."""

    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_mlp, bias=False)
        self.up = nn.Linear(d_model, d_mlp, bias=False)
        self.down = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rope_tables(positions, head_dim: int, theta: float, dtype):
    """cos, sin of position × theta^(-2i/d) for every pair (i, i + d/2) of a head vector. positions: (n,)."""
    inverse = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)  # on the CPU, as transformers does
    angles = positions.float()[:, None] * inverse.to(positions.device)[None]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate(x, cos, sin):  # x: (..., n, d); cos, sin: (n, d)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [heads] Medusa: heads on the last hidden state
class MedusaHeads(nn.Module):
    """Five heads beside Vicuna's LM head, each a residual block x + SiLU(W x + b) and an LM head of its own. Head i
    guesses the token i + 1 places after the one Vicuna predicts, from Vicuna's last hidden state at the anchor
    alone: no head sees another head's guess, so all five are one pass, and a fixed tree combines their top tokens."""

    def __init__(self, c: Vicuna):
        super().__init__()
        self.linear = nn.ModuleList(nn.Linear(c.d_model, c.d_model) for _ in range(c.medusa_heads))
        self.lm_head = nn.ModuleList(nn.Linear(c.d_model, c.vocab_size, bias=False) for _ in range(c.medusa_heads))

    def forward(self, h):  # h: (d_model,), after Vicuna's final RMSNorm → (heads, vocab)
        return torch.stack([out(h + F.silu(block(h))) for block, out in zip(self.linear, self.lm_head)])


# %% [draft] EAGLE: one decoder layer over Vicuna's features
class EagleDraft(nn.Module):
    """fc([e(x_{t+1}); f_t]) → one Llama decoder layer (its attention reads fc's output without an RMSNorm) →
    f̂_{t+1}, a prediction of Vicuna's next last hidden state, which Vicuna's own LM head reads. Fed its own
    predictions and the tokens drawn from them, it drafts autoregressively, one draft forward per tree level."""

    def __init__(self, c: Vicuna):
        super().__init__()
        self.c = c
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)  # a frozen copy of Vicuna's
        self.fc = nn.Linear(2 * c.d_model, c.d_model)
        self.q, self.k, self.v, self.out = (nn.Linear(c.d_model, c.d_model, bias=False) for _ in range(4))
        self.norm = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c.d_model, c.d_mlp)

    def forward(self, features, ids, positions, mask):
        """features: (n, d_model) Vicuna's f_t; ids: (n,) the token one place after each; mask: (n, n) bool."""
        c, n = self.c, ids.shape[0]
        x = self.fc(torch.cat((self.embedding(ids), features), dim=-1))
        cos, sin = rope_tables(positions, c.d_model // c.n_heads, c.rope_theta, x.dtype)
        q, k, v = (proj(x).view(1, n, c.n_heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        o = F.scaled_dot_product_attention(rotate(q, cos, sin), rotate(k, cos, sin), v, attn_mask=mask)
        x = x + self.out(o.transpose(1, 2).reshape(n, -1))
        return x + self.mlp(self.norm(x))


# %% [block] DFlash and DSpark: a block of drafts in one pass
class DraftLayer(nn.Module):
    """A Qwen3 decoder layer whose keys and values also come from the context features c, and whose block slots
    attend to each other without a causal mask."""

    def __init__(self, c: Qwen3):
        super().__init__()
        self.c = c
        self.norm1, self.norm2 = RMSNorm(c.d_model, c.eps), RMSNorm(c.d_model, c.eps)
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(c.head_dim, c.eps), RMSNorm(c.head_dim, c.eps)
        self.mlp = MLP(c.d_model, c.d_mlp)

    def forward(self, x, context, cos, sin):
        """x: (b, d_model) the block; context: (m, d_model) fused features of the positions before it; cos, sin for
        the m context positions, then the block's."""
        c, b, m = self.c, x.shape[0], context.shape[0]
        h = self.norm1(x)
        q = rotate(self.q_norm(self.q(h).view(b, c.n_heads, c.head_dim)).transpose(0, 1), cos[m:], sin[m:])
        k = torch.cat((self.k(context), self.k(h))).view(m + b, c.n_kv_heads, c.head_dim)
        k = rotate(self.k_norm(k).transpose(0, 1), cos, sin)
        v = torch.cat((self.v(context), self.v(h))).view(m + b, c.n_kv_heads, c.head_dim).transpose(0, 1)
        o = F.scaled_dot_product_attention(q[None], k[None], v[None], scale=c.head_dim**-0.5, enable_gqa=True)
        x = x + self.out(o[0].transpose(0, 1).reshape(b, -1))
        return x + self.mlp(self.norm2(x))


class BlockDrafter(nn.Module):
    """fc fuses the five tapped block outputs of every context token into c (12 800 → 2560, then an RMSNorm); five
    draft layers run over the block, the anchor (the last accepted token) and mask tokens, embedded by Qwen3's own
    table, and Qwen3's LM head reads every slot: one pass for the whole block. DFlash (block 16) drafts from the 15
    mask slots; DSpark (block 7) drafts from all 7 slots and adds a Markov head, a left-to-right bias from the token
    drafted just before each slot (W2 · W1[x_{k-1}], rank 256), so its drafts are picked one after another."""

    def __init__(self, c: Qwen3, markov: bool = False):
        super().__init__()
        self.c = c
        self.fc = nn.Linear(len(c.feature_layers) * c.d_model, c.d_model, bias=False)
        self.hidden_norm = RMSNorm(c.d_model, c.eps)
        self.layers = nn.ModuleList(DraftLayer(c) for _ in range(c.draft_layers))
        self.norm = RMSNorm(c.d_model, c.eps)
        if markov:
            self.markov_w1 = nn.Embedding(c.vocab_size, c.markov_rank)
            self.markov_w2 = nn.Linear(c.markov_rank, c.vocab_size, bias=False)

    @torch.no_grad()
    def propose(self, embedding, lm_head, anchor, taps, start, block):
        """anchor: the token at position \`start\`; taps: (m, 5 × d_model) Qwen3's features of positions start - m …
        start - 1. Returns the drafts and their logits."""
        c, device = self.c, taps.device
        ids = torch.full((block,), c.mask, device=device)
        ids[0] = anchor
        context = self.hidden_norm(self.fc(taps))
        positions = torch.arange(start - len(taps), start + block, device=device)
        cos, sin = rope_tables(positions, c.head_dim, c.rope_theta, taps.dtype)
        x = embedding(ids)
        for layer in self.layers:
            x = layer(x, context, cos, sin)
        logits = lm_head(self.norm(x))
        if not hasattr(self, "markov_w1"):  # DFlash: slot k predicts the token in slot k
            logits = logits[1:]
            return logits.argmax(-1), logits
        drafts, corrected, previous = [], [], ids[:1]  # DSpark: slot k predicts the token k + 1 after the anchor
        for k in range(block):
            z = logits[k : k + 1] + self.markov_w2(self.markov_w1(previous))
            previous = z.argmax(-1)
            drafts.append(previous)
            corrected.append(z)
        return torch.cat(drafts), torch.cat(corrected)


# %% [verify] One acceptance rule for trees and blocks
def accept_tree(tokens, parents, argmax):
    """Medusa and EAGLE: along each root-to-leaf path keep tokens while each equals the target's argmax at its parent;
    the longest kept path wins (the first leaf on ties). Returns the kept nodes after the root."""
    is_parent, best = set(parents), []
    for leaf in range(len(parents)):
        if leaf in is_parent:
            continue
        path = [leaf]
        while parents[path[0]] >= 0:
            path.insert(0, parents[path[0]])
        kept = 0
        while kept + 1 < len(path) and tokens[path[kept + 1]] == argmax[path[kept]]:
            kept += 1
        if kept > len(best):
            best = path[1 : kept + 1]
    return best


def accept_block(drafts, argmax, sent):
    """DFlash and DSpark: the first \`sent\` drafts went to the target in order; keep them while each equals the
    target's argmax at the slot before it. argmax[k] is the target's choice after the anchor and k drafts."""
    kept = 0
    while kept < sent and drafts[kept] == argmax[kept]:
        kept += 1
    return kept


# %% [acceptance] Acceptance length and target forwards
def acceptance(accepted, cap):
    """accepted: drafts kept at each verification forward. Each forward settles accepted + 1 tokens (the kept drafts
    and the target's own next token), at most \`cap\`. Returns the mean acceptance length τ, how many forwards settled
    1 … cap tokens, and the target forwards for the whole answer (one more for the prompt). Plain greedy settles 1
    per forward, so τ is the most a drafter can speed decoding up while drafting and the larger verification
    forward cost nothing."""
    settled = torch.as_tensor(accepted) + 1
    histogram = torch.bincount(settled - 1, minlength=cap)
    return settled.float().mean(), histogram, len(accepted) + 1


# %% [cost] Extra parameters
def extra_parameters(state, shared=()):
    """Parameters a drafter adds to its target: every tensor of its checkpoint except the ones that are copies of
    the target's (EAGLE's embedding, DSpark's embedding and LM head); Medusa's heads have LM heads of their own."""
    return sum(value.numel() for name, value in state.items() if not any(name.startswith(s) for s in shared))
`;export{e as default};
