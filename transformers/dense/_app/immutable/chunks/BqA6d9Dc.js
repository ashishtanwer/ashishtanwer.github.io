const e=`# The refusal direction of Qwen2.5-0.5B-Instruct (Arditi et al. 2024, "Refusal in Language Models Is Mediated by a
# Single Direction", arXiv:2406.11717): the mean residual on harmful instructions minus the mean on harmless ones,
# taken where the instruction ends. Removing that one direction everywhere stops the model refusing; adding it at one
# layer makes it refuse harmless requests.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    layers: int = 24
    hidden: int = 896
    eoi_tokens: int = 5  # after the instruction: <|im_end|> \\n <|im_start|> assistant \\n
    refusal_tokens: tuple = (40, 2121)  # 'I', 'As': how this model's refusals start
    max_kl: float = 0.2  # the paper's 0.1 leaves no candidate on this 0.5B model
    max_depth: float = 0.8
    new_tokens: int = 64


# %% [means] Candidates: harmful mean − harmless mean
def block_inputs(model, ids):
    """(layers, tokens, hidden): the residual stream entering each block."""
    seen = []
    handles = [block.register_forward_pre_hook(lambda module, args: seen.append(args[0][0])) for block in
               model.model.layers]
    with torch.no_grad():
        model(ids[None])
    for handle in handles:
        handle.remove()
    return torch.stack(seen)


def candidate_directions(model, harmful_ids, harmless_ids, eoi=5):
    """(eoi positions, layers, hidden): one candidate per layer and post-instruction position."""
    def mean(prompts):
        return torch.stack([block_inputs(model, ids)[:, -eoi:] for ids in prompts]).double().mean(0)

    return (mean(harmful_ids) - mean(harmless_ids)).transpose(0, 1)


# %% [refusal-score] How strongly the model starts to refuse
def refusal_score(logits, refusal_tokens=(40, 2121)):
    """log p − log(1 − p), p = the probability that the first answer token is a refusal token."""
    p = logits.double().softmax(-1)[..., list(refusal_tokens)].sum(-1)
    return p.log() - (1 - p).log()


# %% [ablate] Directional ablation: remove the direction everywhere
def ablation_hooks(model, direction):
    """x ← x − (x·r̂) r̂ on every block's input and on every attention and MLP output, so no layer can write the
    direction into the residual stream or read it from there. Returns the hook handles."""
    r = direction / direction.norm()
    drop = lambda x: x - (x @ r)[..., None] * r  # noqa: E731
    handles = []
    for block in model.model.layers:
        handles.append(block.register_forward_pre_hook(lambda module, args: (drop(args[0]),) + args[1:]))
        handles.append(block.self_attn.register_forward_hook(lambda module, inp, out: (drop(out[0]),) + out[1:]))
        handles.append(block.mlp.register_forward_hook(lambda module, inp, out: drop(out)))
    return handles


# %% [orthogonalize] The same edit, made in the weights
def orthogonalize(model, direction):
    """W ← W − r̂ r̂ᵀ W for every matrix that writes to the residual stream (the token embeddings, each
    attention output projection and each MLP down projection): the edited model behaves like the ablated one with
    no hooks.
    Qwen2.5-0.5B ties its unembedding to the embeddings, so they are untied first."""
    r = (direction / direction.norm()).to(model.dtype)
    with torch.no_grad():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        embed = model.model.embed_tokens.weight
        embed -= (embed @ r)[:, None] * r  # rows are the vectors each token writes
        for block in model.model.layers:
            for linear in (block.self_attn.o_proj, block.mlp.down_proj):
                linear.weight -= r[:, None] * (r @ linear.weight)[None]  # weight: (out = hidden, in)
    return model


# %% [add] Activation addition: add the direction at one layer
def addition_hook(model, direction, layer):
    """x ← x + r at the input of one block, at every position (prompt and answer alike)."""
    return [model.model.layers[layer].register_forward_pre_hook(lambda module, args: (args[0] + direction,) + args[1:])]


# %% [select] Choosing one candidate
def select(bypass, induce, kl, max_kl=0.2, max_depth=0.8):
    """Lowest bypass score (refusal left on harmful prompts once ablated) among candidates that still induce refusal
    when added (induce > 0), barely change harmless predictions when ablated (KL < max_kl), and are not too deep.
    A zero candidate (layer 0, where every prompt has the same template tokens) has no direction: its scores are NaN."""
    positions, layers = bypass.shape
    allowed = bypass.isfinite() & (induce > 0) & (kl < max_kl) & (torch.arange(layers)[None] < max_depth * layers)
    best = torch.where(allowed, bypass, torch.full_like(bypass, float("inf"))).flatten().argmin()
    return divmod(int(best), layers)  # (position from the first post-instruction token, layer)


# %% [generate] Greedy answers under an intervention
def greedy(model, ids, handles_for=None, new_tokens=64, stop=(151645, 151643)):
    """Argmax decoding token by token, with the intervention's hooks installed for the whole answer."""
    handles = handles_for(model) if handles_for else []
    answer = []
    try:
        with torch.no_grad():
            for _ in range(new_tokens):
                logits = model(torch.cat([ids, torch.tensor(answer, dtype=ids.dtype)])[None]).logits[0, -1]
                token = int(logits.argmax())
                answer.append(token)
                if token in stop:
                    break
    finally:
        for handle in handles:
            handle.remove()
    return answer


# %% [model] The whole method
def refusal_direction(model, train, val, cfg=Config()):
    """train / val: {'harmful': [ids], 'harmless': [ids]} chat-templated prompts. Returns the chosen direction and
    its (position, layer)."""
    candidates = candidate_directions(model, train["harmful"], train["harmless"], cfg.eoi_tokens)
    score = lambda ids: refusal_score(model(ids[None]).logits[0, -1])  # noqa: E731
    bypass, induce, kl = (torch.zeros(cfg.eoi_tokens, cfg.layers) for _ in range(3))
    with torch.no_grad():
        base = [model(ids[None]).logits[0, -1].double().log_softmax(-1) for ids in val["harmless"]]
        for pos in range(cfg.eoi_tokens):
            for layer in range(cfg.layers):
                r = candidates[pos, layer].float()
                handles = ablation_hooks(model, r)
                bypass[pos, layer] = torch.stack([score(ids) for ids in val["harmful"]]).mean()
                kl[pos, layer] = torch.stack([
                    (b.exp() * (b - model(ids[None]).logits[0, -1].double().log_softmax(-1))).sum()
                    for ids, b in zip(val["harmless"], base)]).mean()
                for handle in handles:
                    handle.remove()
                handles = addition_hook(model, r, layer)
                induce[pos, layer] = torch.stack([score(ids) for ids in val["harmless"]]).mean()
                for handle in handles:
                    handle.remove()
    pos, layer = select(bypass, induce, kl, cfg.max_kl, cfg.max_depth)
    return candidates[pos, layer], pos, layer
`;export{e as default};
