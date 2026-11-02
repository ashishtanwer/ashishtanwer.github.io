const e=`# Attribution graphs on Gemma 2 2B, the circuits tab, as plain PyTorch (Ameisen et al. 2025, "Circuit Tracing").
# Replace every MLP by its transcoder's active features plus an error node (what the transcoder missed), and freeze the
# attention patterns and the norms' scales at their values on this prompt. What is left is linear in the nodes' writes
# to the residual stream, so the direct effect of a source on a target is a gradient: inject the target's read vector
# (a feature's encoder row, or a logit's unembedding row minus the mean row), run it back through the frozen attention
# layers, and dot it with the source's write (activation × decoder row, an error vector, a token embedding). Pruning
# keeps the nodes and edges with the most influence on the output; steering adds a feature's decoder row, scaled, to
# its MLP output in the real model. Weights: google/gemma-2-2b; transcoders: google/gemma-scope-2b-pt-transcoders.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    layers: int = 26
    d_model: int = 2304
    heads: int = 8
    kv_heads: int = 4
    d_head: int = 256
    rms_eps: float = 1e-6
    query_scale: float = 256**-0.5  # query_pre_attn_scalar
    attn_softcap: float = 50.0
    final_softcap: float = 30.0
    rope_theta: float = 10000.0


# %% [block] A Gemma 2 block, keeping what the replacement model freezes
def rms_scale(x, c: Config = Config()):
    """1 / rms(x) per position: the part of RMSNorm the replacement model freezes."""
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c.rms_eps)


def rope(x, c: Config = Config()):
    """x (heads, T, d_head) rotated by position (rotate-half layout)."""
    T = x.shape[1]
    inv = 1.0 / c.rope_theta ** (torch.arange(0, c.d_head, 2, dtype=torch.float32) / c.d_head)
    angles = torch.arange(T, dtype=torch.float32)[:, None] * inv  # (T, d_head / 2)
    cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin


def block(layer, h, c: Config = Config()):
    """One Gemma 2 decoder layer on h (T, d): the output, and the cache the attribution needs: the norm scales of the
    attention input (s_in), attention output (s_attn), MLP input (s_mlp), the attention pattern per head, the MLP
    input as the transcoder reads it (x / rms(x), no gain) and the MLP output after its norm."""
    T = h.shape[0]
    s_in = rms_scale(h, c)
    x = h * s_in * (1 + layer.input_layernorm.weight)
    a = layer.self_attn
    q = (x @ a.q_proj.weight.T).view(T, c.heads, c.d_head).transpose(0, 1)
    k = (x @ a.k_proj.weight.T).view(T, c.kv_heads, c.d_head).transpose(0, 1)
    v = (x @ a.v_proj.weight.T).view(T, c.kv_heads, c.d_head).transpose(0, 1)
    q, k = rope(q, c), rope(k, c)
    group = c.heads // c.kv_heads
    k, v = k.repeat_interleave(group, 0), v.repeat_interleave(group, 0)
    scores = q @ k.transpose(-1, -2) * c.query_scale
    scores = torch.tanh(scores / c.attn_softcap) * c.attn_softcap
    causal = torch.ones(T, T, dtype=torch.bool).tril()  # the sliding window (4096) never binds on a short prompt
    pattern = scores.masked_fill(~causal, float("-inf")).softmax(-1)
    z = (pattern @ v).transpose(0, 1).reshape(T, c.heads * c.d_head)
    o = z @ a.o_proj.weight.T
    s_attn = rms_scale(o, c)
    mid = h + o * s_attn * (1 + layer.post_attention_layernorm.weight)
    s_mlp = rms_scale(mid, c)
    x_norm = mid * s_mlp  # what the transcoder reads
    m = layer.mlp
    y = (F.gelu(x_norm * (1 + layer.pre_feedforward_layernorm.weight) @ m.gate_proj.weight.T, approximate="tanh")
         * (x_norm * (1 + layer.pre_feedforward_layernorm.weight) @ m.up_proj.weight.T)) @ m.down_proj.weight.T
    s_y = rms_scale(y, c)
    mlp_out = y * s_y * (1 + layer.post_feedforward_layernorm.weight)
    cache = {"s_in": s_in, "s_attn": s_attn, "s_mlp": s_mlp, "pattern": pattern, "x_norm": x_norm,
             "mlp_out": mlp_out, "mid": mid}
    return mid + mlp_out, cache


def forward(model, ids, c: Config = Config()):
    """ids (T,) → the logits (after Gemma 2's final softcap), the final norm scale and every layer's cache."""
    g = model.model
    h = g.embed_tokens.weight[ids] * torch.tensor(c.d_model**0.5)
    caches = []
    for layer in g.layers:
        h, cache = block(layer, h, c)
        caches.append(cache)
    s_final = rms_scale(h, c)
    logits = (h * s_final * (1 + g.norm.weight)) @ model.lm_head.weight.T
    return torch.tanh(logits / c.final_softcap) * c.final_softcap, s_final, caches, h


# %% [transcoder] A transcoder in place of an MLP
def encode(tc, x_norm):
    """JumpReLU features of x / rms(x): on only where the pre-activation clears the feature's own threshold."""
    pre = x_norm @ tc["W_enc"] + tc["b_enc"]
    return pre * (pre > tc["threshold"])


def decode(tc, acts):
    return acts @ tc["W_dec"] + tc["b_dec"]


# %% [replacement] The local replacement model: nodes and what is frozen
def nodes(caches, transcoders):
    """Every node's write to the stream: active features (layer, position, feature, activation; position 0, <bos>,
    left out), and per layer the error vector the transcoder leaves, mlp_out − decode(acts) (zero at <bos>)."""
    features, errors = [], []
    for layer, (cache, tc) in enumerate(zip(caches, transcoders)):
        acts = encode(tc, cache["x_norm"])
        acts[0] = 0
        for pos, f in acts.nonzero().tolist():
            features.append((layer, pos, f, acts[pos, f]))
        err = cache["mlp_out"] - decode(tc, acts)
        err[0] = 0
        errors.append(err)
    return features, torch.stack(errors)


def frozen_attention_back(layer, cache, g, c: Config = Config()):
    """The transpose of one attention sublayer with its pattern and norm scales frozen: g (B, T, d) at its output
    (the stream after it) → g at its input, plus the part through the residual connection."""
    a = layer.self_attn
    B, T, _ = g.shape
    g_o = g * cache["s_attn"] * (1 + layer.post_attention_layernorm.weight)
    g_z = (g_o @ a.o_proj.weight).view(B, T, c.heads, c.d_head).transpose(1, 2)  # (B, heads, T, d_head)
    g_v = cache["pattern"].transpose(-1, -2) @ g_z  # (B, heads, T, d_head): Pᵀ g, per query head
    g_v = g_v.view(B, c.kv_heads, c.heads // c.kv_heads, T, c.d_head).sum(2).transpose(1, 2).reshape(B, T, -1)
    g_x = g_v @ a.v_proj.weight
    return g + g_x * cache["s_in"] * (1 + layer.input_layernorm.weight)


# %% [attribution] Edges: one backward through the frozen model per target
def edges_into(model, caches, top_layer, seeds, features, errors, ids, c: Config = Config()):
    """seeds (B, T, d): each target's read vector placed at its position in the stream entering MLP \`top_layer\` (or,
    for logits, the final stream: top_layer = layers). Returns (B, sources): the direct effect of every feature, error
    and token node on each target, in circuit-tracer's node order (features, errors layer by layer, tokens)."""
    layers = model.model.layers
    T = ids.shape[0]
    by_layer = {}
    for i, (layer, pos, f, act) in enumerate(features):
        by_layer.setdefault(layer, []).append((i, pos, f, act))
    out = torch.zeros(seeds.shape[0], len(features) + (c.layers + 1) * T)
    # a feature target reads the stream after its own layer's attention: undo that attention first
    g = seeds if top_layer == c.layers else frozen_attention_back(layers[top_layer], caches[top_layer], seeds, c)
    for layer in reversed(range(top_layer)):
        # g is the gradient at the output of block \`layer\`: what its MLP writes is read with it
        for i, pos, f, act in by_layer.get(layer, []):
            out[:, i] = g[:, pos] @ (act * model.transcoders[layer]["W_dec"][f])
        out[:, len(features) + layer * T: len(features) + (layer + 1) * T] = (g * errors[layer]).sum(-1)
        g = frozen_attention_back(layers[layer], caches[layer], g, c)  # the MLP passes nothing back
    embed = model.model.embed_tokens.weight[ids] * torch.tensor(c.d_model**0.5)
    out[:, len(features) + c.layers * T:] = (g * embed).sum(-1)
    return out


def feature_seeds(features, targets, caches, model, T, c: Config = Config()):
    """A feature target's read vector at the stream entering its MLP: W_enc[:, f] · s_mlp (the frozen 1 / rms)."""
    seeds = torch.zeros(len(targets), T, c.d_model)
    for b, i in enumerate(targets):
        layer, pos, f, _ = features[i]
        seeds[b, pos] = model.transcoders[layer]["W_enc"][:, f] * caches[layer]["s_mlp"][pos]
    return seeds


def logit_seeds(model, tokens, s_final, T, c: Config = Config()):
    """A logit target's read vector at the final stream: (W_U[token] − mean row) through the frozen final norm."""
    W_U = model.lm_head.weight
    rows = W_U[tokens] - W_U.mean(0)
    seeds = torch.zeros(len(tokens), T, c.d_model)
    seeds[:, -1] = rows * (1 + model.model.norm.weight) * s_final[-1]
    return seeds


# %% [pruning] Influence and pruning
def normalize(A):
    """Each target's row of |direct effects|, summing to 1: the share each source has in it."""
    A = A.abs()
    return A / A.sum(1, keepdim=True).clamp(min=1e-10)


def influence(A, logit_weights):
    """Total influence on the output, every path at once: w (A + A² + …), w = the logits' probabilities."""
    current = logit_weights @ A
    total = current.clone()
    while current.any():
        current = current @ A
        total += current
    return total


def threshold_for(scores, keep):
    """The score above which the largest scores add up to \`keep\` of the total."""
    ordered = scores.sort(descending=True).values
    cumulative = ordered.cumsum(0) / ordered.sum()
    index = min(int(torch.searchsorted(cumulative, keep)), max(int((ordered > 0).sum()) - 1, 0))
    return ordered[index]


def prune(A, logit_p, n_features, n_tokens, node_keep=0.8, edge_keep=0.98):
    """circuit-tracer's prune_graph: keep the nodes with \`node_keep\` of the influence, then the edges with
    \`edge_keep\` of the edge influence among them, then drop nodes left without edges. Tokens and logits stay."""
    n_logits = len(logit_p)
    w = torch.zeros(A.shape[0])
    w[-n_logits:] = logit_p
    node_score = influence(normalize(A), w)
    node_mask = node_score >= threshold_for(node_score, node_keep)
    node_mask[-n_logits - n_tokens:] = True
    pruned = A.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0
    norm = normalize(pruned)
    edge_score = norm * (influence(norm, w) + w)[:, None]
    edge_mask = edge_score >= threshold_for(edge_score.flatten(), edge_keep)
    while True:
        old = node_mask.clone()
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        node_mask[:n_features] &= edge_mask[:n_features].any(1)
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        if torch.equal(old, node_mask):
            return node_mask, edge_mask, node_score


# %% [intervention] Steering the real model with features
@torch.no_grad()
def steer(model, ids, deltas, c: Config = Config()):
    """The logits at the last position with vectors added to MLP outputs: deltas = [(layer, pos, vector)], for a
    feature set to m × its activation a: (m − 1) · a · W_dec[f]. Nothing is frozen; the model runs on."""
    g = model.model
    h = g.embed_tokens.weight[ids] * torch.tensor(c.d_model**0.5)
    for i, layer in enumerate(g.layers):
        out, cache = block(layer, h, c)
        for layer_, pos, vec in deltas:
            if layer_ == i:
                out[pos] += vec
        h = out
    logits = (h[-1] * rms_scale(h[-1], c) * (1 + g.norm.weight)) @ model.lm_head.weight.T
    return torch.tanh(logits / c.final_softcap) * c.final_softcap


# %% [model] Wiring: the graph of a prompt
def attribution_graph(model, ids, logit_tokens, logit_p, batch=256, c: Config = Config()):
    """The full adjacency matrix A[target, source] in circuit-tracer's node order: features, errors, tokens, logits.
    model.transcoders: one dict per layer (W_enc, b_enc, threshold, W_dec, b_dec)."""
    with torch.no_grad():
        _, s_final, caches, _ = forward(model, ids, c)
        features, errors = nodes(caches, model.transcoders)
        T = ids.shape[0]
        n = len(features) + (c.layers + 1) * T + len(logit_tokens)
        A = torch.zeros(n, n)
        rows = edges_into(model, caches, c.layers, logit_seeds(model, logit_tokens, s_final, T, c), features, errors,
                          ids, c)
        A[-len(logit_tokens):, : rows.shape[1]] = rows
        for layer in range(c.layers):
            targets = [i for i, f in enumerate(features) if f[0] == layer]
            for start in range(0, len(targets), batch):
                chunk = targets[start:start + batch]
                seeds = feature_seeds(features, chunk, caches, model, T, c)
                A[chunk, : rows.shape[1]] = edges_into(model, caches, layer, seeds, features, errors, ids, c)
    return A, features


# ids = tokenizer("Fact: the capital of the state containing Dallas is", return_tensors="pt").input_ids[0]
# model.transcoders = [load(layer) for layer in range(26)]  # Gemma Scope, circuit-tracer's L0 per layer
# A, features = attribution_graph(model, ids, logit_tokens, logit_p)
# node_mask, edge_mask, score = prune(A, logit_p, len(features), len(ids))
`;export{e as default};
