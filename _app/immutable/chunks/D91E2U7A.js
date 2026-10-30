const e=`# Wanda (Sun, Liu, Bair & Kolter 2023, arXiv:2306.11695) pruning Qwen2.5-0.5B-Instruct: each weight is scored by
# |W_ij| times the norm of its input channel over calibration tokens, and the lowest-scoring half of every output
# row is set to zero (or the two lowest of every four consecutive inputs, for 2:4). One pass, no weight update.
from dataclasses import dataclass

import torch


@dataclass
class Config:
    layers: int = 24
    hidden: int = 896  # input width of q, k, v, gate and up
    intermediate: int = 4864  # input width of down
    kv_width: int = 128  # rows of k_proj and v_proj: 2 key-value heads × 64
    sparsity: float = 0.5
    n: int = 2  # 2:4 sparsity: prune n of every m consecutive inputs of a row
    m: int = 4
    calibration_sequences: int = 128  # windows of C4 text
    sequence_length: int = 2048


LINEARS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


# %% [calibration] Input statistics from calibration tokens
class InputStats:
    """Squared norm of every input channel over the calibration tokens, per sequence (Wanda's scaler_row):
    ‖X_j‖₂ = √(sequences · scaler_row_j)."""

    def __init__(self, in_features):
        self.sum_squares = torch.zeros(in_features)
        self.sequences = 0

    def add(self, x):  # x: (batch, tokens, in_features), the linear layer's input
        self.sum_squares += x.float().pow(2).sum(dim=(0, 1))
        self.sequences += x.shape[0]

    @property
    def scaler_row(self):
        return self.sum_squares / self.sequences


# %% [score] The Wanda score |W|·‖X‖
def wanda_score(weight, scaler_row):
    # weight: (out_features, in_features); a weight on a loud input channel scores high even when it is small
    return weight.abs() * scaler_row.sqrt()[None, :]


# %% [mask] Keep half of every row, or 2 of every 4
def prune_mask(score, sparsity=0.5, n=0, m=0):
    """True where a weight is pruned. Weights compete only within their own output row."""
    if n:  # n:m — inside every group of m consecutive inputs, the n lowest scores go
        groups = score.reshape(score.shape[0], -1, m)  # (out, in / m, m)
        lowest = groups.topk(n, dim=-1, largest=False).indices
        return torch.zeros_like(groups, dtype=torch.bool).scatter_(-1, lowest, True).reshape(score.shape)
    order = score.sort(dim=-1, stable=True).indices  # lowest first
    pruned = order[:, : int(score.shape[1] * sparsity)]
    return torch.zeros_like(score, dtype=torch.bool).scatter_(1, pruned, True)


# %% [magnitude] The baseline: magnitude pruning
def magnitude_mask(weight, sparsity=0.5, n=0, m=0):
    """The Wanda repository's baseline: one |W| threshold for the whole matrix (so rows keep different counts);
    for n:m the n smallest |W| of every group."""
    if n:
        return prune_mask(weight.abs(), n=n, m=m)
    magnitude = weight.abs()
    threshold = magnitude.flatten().sort().values[int(weight.numel() * sparsity)]
    return magnitude <= threshold


# %% [error] How much the layer's output moves
def output_error(weight, pruned, hessian):
    """‖(W − Ŵ)X‖² / ‖WX‖² over the calibration inputs X, from H = 2XXᵀ/n (the scale cancels)."""
    delta = (weight - pruned).double()
    weight, hessian = weight.double(), hessian.double()
    return ((delta @ hessian) * delta).sum() / ((weight @ hessian) * weight).sum()


# %% [model] Pruning the whole model, layer by layer
class StopForward(Exception):
    pass


def layer_inputs(model, ids):
    """The hidden states entering decoder layer 0 for each calibration sequence, and the mask and rotary
    arguments every decoder layer is called with."""
    hidden, kwargs = [], {}

    def grab(module, args, call_kwargs):
        hidden.append(args[0])
        kwargs.update({k: call_kwargs[k] for k in ("attention_mask", "position_ids", "position_embeddings")})
        raise StopForward

    handle = model.model.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    for sequence in ids:
        try:
            model(sequence[None], use_cache=False)
        except StopForward:
            pass
    handle.remove()
    return hidden, kwargs


@torch.no_grad()
def prune_model(model, ids, sparsity=0.5, n=0, m=0):
    """ids: (sequences, tokens) of calibration text. Each layer is pruned from the inputs that the already
    pruned layers before it produce; its pruned outputs become the next layer's inputs."""
    hidden, kwargs = layer_inputs(model, ids)
    for layer in model.model.layers:
        stats = {name: InputStats(layer.get_submodule(name).in_features) for name in LINEARS}
        hooks = [
            layer.get_submodule(name).register_forward_hook(lambda mod, inp, out, s=stats[name]: s.add(inp[0]))
            for name in LINEARS
        ]
        for x in hidden:
            layer(x, **kwargs)
        for hook in hooks:
            hook.remove()
        for name in LINEARS:
            linear = layer.get_submodule(name)
            score = wanda_score(linear.weight, stats[name].scaler_row)
            linear.weight[prune_mask(score, sparsity, n, m)] = 0
        hidden = [layer(x, **kwargs) for x in hidden]
    return model
`;export{e as default};
