const e=`# ZeRO, the Zero Redundancy Optimizer, as DeepSpeed runs it, for GPT-2 in the Parallelism Explainer's ZeRO tab, as
# plain PyTorch. Data parallel training keeps every model state on every rank; stage 1 shards AdamW's state among the
# ranks, stage 2 the gradients too, stage 3 the weights too. The ranks are lists in one process; the recorded run
# used DeepSpeed on four processes over gloo, in float32. Each "# %% [id]" section is one block.
from dataclasses import dataclass

import torch


@dataclass
class Config:  # the recorded run
    ranks: int = 4
    sequences_per_rank: int = 2
    bytes_per_value: int = 4  # float32
    alignment: int = 2  # stages 1–2 pad the flat buffer to a multiple of alignment × ranks values
    persistence_threshold: int = 100_000  # stage 3 keeps tensors under this many values whole on every rank
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


# %% [partition] Partitions
def flat_partition(sizes, cfg):
    """Stages 1–2: every tensor laid end to end in one flat buffer, padded to a multiple of alignment × ranks and cut
    into equal parts; rank r owns values [r·P, (r+1)·P). The cuts fall wherever they fall, inside tensors."""
    total = sum(sizes)
    padding = -total % (cfg.alignment * cfg.ranks)
    return (total + padding) // cfg.ranks, padding


def tensor_partition(numel, cfg):
    """Stage 3: each tensor padded to a multiple of the ranks and cut into equal parts, so any one of them can be
    gathered alone; tensors under the persistence threshold (biases, LayerNorms) stay whole on every rank."""
    return -(-numel // cfg.ranks), numel < cfg.persistence_threshold


# %% [stages] What each stage reduces and gathers
def all_reduce_mean(grads):
    """Stages 0–1: every rank gets the average of all ranks' full gradients."""
    mean = torch.stack(grads).mean(0)
    return [mean.clone() for _ in grads]


def reduce_scatter_mean(grads, ranks):
    """Stages 2–3: rank r gets the average of part r only. In the recorded run DeepSpeed's stages 1 and 2 get there by
    all-reducing each bucket and keeping their part (use_multi_rank_bucket_allreduce); stage 3 reduce-scatters."""
    return list(torch.stack(grads).mean(0).chunk(ranks))


def all_gather(parts):
    """Stages 1–2 after the step, stage 3 before each use: every rank gets every rank's updated part."""
    return torch.cat(parts)


# %% [memory] Memory per rank
def rank_bytes(sizes, stage, cfg):
    """Model states one rank holds in float32. Stage 0: everything. Stages 1–2: full weights, plus DeepSpeed's float32
    copy of the rank's part (the copy AdamW updates) and AdamW's two moments of it. Stage 3: its part of every tensor
    (updated in place, no copy) and the whole of each small persistent one, and AdamW's moments of its part."""
    b = cfg.bytes_per_value
    total = sum(sizes)
    if stage == 0:
        return {"weights": total * b, "master": 0, "optimizer": 2 * total * b}
    if stage in (1, 2):
        part, _ = flat_partition(sizes, cfg)
        return {"weights": total * b, "master": part * b, "optimizer": 2 * part * b}
    parts = [tensor_partition(n, cfg) for n in sizes]
    owned = sum(p for p, _ in parts)
    whole = sum(n for n, (_, keep) in zip(sizes, parts) if keep)
    return {"weights": (owned + whole) * b, "master": 0, "optimizer": 2 * owned * b}


# %% [step] One ZeRO step
def zero_step(stage, replicas, parts, optimizers, batch, loss_fn, cfg):
    """One step at any stage. \`parts[r]\` is rank r's flat nn.Parameter (the whole model at stage 0, its partition
    otherwise) and \`optimizers[r]\` its AdamW. Stage 3 gathers the weights before computing (here all at once; in the
    recorded step 1 DeepSpeed gathers each module just before it runs and frees it after, and from step 2 its
    prefetcher has every weight gathered early in forward); stages 1–2 gather them after the step.
    In float64 this step matches one process on the whole batch to float64 rounding (the check). In float32 it
    differs only by the order of the sums across ranks, which AdamW's division by √v̂ + ε magnifies where a
    gradient is near zero; the recorded run's differences are the size of one process's own when it adds the
    four micro-batches' gradients in turn instead of the whole batch at once."""
    ranks = cfg.ranks
    n = sum(p.numel() for p in replicas[0].parameters())

    def load(flats):
        for model, flat in zip(replicas, flats):
            torch.nn.utils.vector_to_parameters(flat[:n], model.parameters())

    if stage == 3:
        with torch.no_grad():
            load([all_gather([p.detach() for p in parts])] * ranks)
    grads = []
    for model, x in zip(replicas, batch.chunk(ranks)):
        model.zero_grad(set_to_none=True)
        loss_fn(model, x).backward()
        flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        grads.append(torch.cat([flat, flat.new_zeros(parts[0].numel() * (ranks if stage else 1) - n)]))
    reduced = all_reduce_mean(grads) if stage < 2 else reduce_scatter_mean(grads, ranks)
    for r in range(ranks):
        g = reduced[r]
        if stage == 1:
            g = g.chunk(ranks)[r]  # stage 1 reduces everything, then keeps only its part
        parts[r].grad = g.clone()
        optimizers[r].step()
    with torch.no_grad():
        if stage == 0:
            load([p.detach() for p in parts])  # every rank updated its own full copy
        elif stage < 3:
            load([all_gather([p.detach() for p in parts])] * ranks)
`;export{e as default};
