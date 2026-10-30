const e=`# SparseGPT (Frantar & Alistarh 2023, arXiv:2301.00774) pruning Qwen2.5-0.5B-Instruct: each linear layer is pruned
# column by column with the inverse Hessian of its calibration inputs, and every pruned column's error is pushed
# onto the columns not yet visited (an OBS update), so the kept weights make up for the dropped ones.
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
    blocksize: int = 128  # columns whose mask is chosen together (50 %)
    percdamp: float = 0.01  # damping added to H's diagonal, as a fraction of its mean
    calibration_sequences: int = 128
    sequence_length: int = 2048


LINEARS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


# %% [hessian] The Hessian of the layer's reconstruction error
class HessianStats:
    """H = 2XXᵀ/n over the calibration tokens X (n = sequences): the Hessian of ‖WX − ŴX‖² in each row of Ŵ.
    Its diagonal is twice Wanda's per-channel statistic."""

    def __init__(self, in_features):
        self.sum_outer = torch.zeros(in_features, in_features)
        self.sequences = 0

    def add(self, x):  # x: (batch, tokens, in_features), the linear layer's input
        self.sequences += x.shape[0]
        x = x.float().reshape(-1, x.shape[-1])
        self.sum_outer += x.T @ x

    @property
    def H(self):
        return 2 * self.sum_outer / self.sequences


# %% [inverse] H⁻¹ through Cholesky
def inverse_factor(H, percdamp=0.01):
    """Upper Cholesky factor U of (H + λI)⁻¹ (UᵀU = (H + λI)⁻¹): row c of U, from c on, is how column c's error
    spreads over the columns after it once the columns before c are fixed; U[c, c] scales column c's error."""
    H = H.clone()
    dead = torch.diag(H) == 0  # inputs that were always zero
    H[dead, dead] = 1
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(H.shape[0])
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)


# %% [prune] Column by column, with the OBS update
@torch.no_grad()
def sparsegpt_prune(weight, H, sparsity=0.5, n=0, m=0, blocksize=128, percdamp=0.01):
    W = weight.clone().float()  # (out_features, in_features)
    W[:, torch.diag(H) == 0] = 0
    U = inverse_factor(H, percdamp)
    for start in range(0, W.shape[1], blocksize):
        end = min(start + blocksize, W.shape[1])
        block, U1 = W[:, start:end].clone(), U[start:end, start:end]
        final, errors = torch.zeros_like(block), torch.zeros_like(block)
        if n:
            pruned = torch.zeros_like(block, dtype=torch.bool)  # chosen as the sweep reaches each group
        else:  # 50 %: the smallest w² / U_cc² of the whole block, across all rows at once
            score = block**2 / torch.diag(U1).reshape((1, -1)) ** 2
            pruned = score <= torch.sort(score.flatten())[0][int(score.numel() * sparsity)]
        for i in range(end - start):
            if n and i % m == 0:  # n:m — the group's n lowest scores, from the already updated weights
                score = block[:, i:(i + m)] ** 2 / torch.diag(U1)[i:(i + m)].reshape((1, -1)) ** 2
                pruned.scatter_(1, i + torch.topk(score, n, dim=1, largest=False)[1], True)
            w = block[:, i]
            q = w.clone()
            q[pruned[:, i]] = 0
            final[:, i] = q
            error = (w - q) / U1[i, i]
            block[:, i:] -= error.unsqueeze(1).matmul(U1[i, i:].unsqueeze(0))  # OBS: later columns compensate
            errors[:, i] = error
        W[:, start:end] = final
        W[:, end:] -= errors.matmul(U[start:end, end:])  # the block's errors reach the columns after it at once
    return W


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
        stats = {name: HessianStats(layer.get_submodule(name).in_features) for name in LINEARS}
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
            linear.weight.copy_(sparsegpt_prune(linear.weight, stats[name].H, sparsity, n, m))
        hidden = [layer(x, **kwargs) for x in hidden]
    return model
`;export{e as default};
