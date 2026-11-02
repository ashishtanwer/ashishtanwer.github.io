const e=`# Fully sharded data parallelism as PyTorch's FSDP2 (fully_shard) does it, for GPT-2 in the Parallelism Explainer's
# FSDP tab, as plain PyTorch. Every GPT-2 block is a unit and the root unit holds W_E (also the LM head), W_P and
# ln_f. Each rank keeps a quarter of the rows of every tensor, their gradients and their AdamW state; a unit's
# weights are all-gathered just before it computes and freed after, and its gradients are reduce-scattered after its
# backward. The ranks are lists in one process; the recorded run used four processes over gloo. Each "# %% [id]"
# section is one block.
from dataclasses import dataclass

import torch


@dataclass
class Config:  # the recorded run
    ranks: int = 4
    sequences_per_rank: int = 2
    sequence_length: int = 128
    bytes_per_value: int = 4  # float32
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


# %% [shard] Rows on dim 0
def shard(tensor, ranks):
    """FSDP2's Shard(0): pad dim 0 up to a multiple of the ranks and cut it into equal chunks; rank r keeps chunk r.
    W_E's 50,257 rows become 4 × 12,565: the last rank holds 12,562 real rows and 3 rows of padding."""
    rows = -(-tensor.shape[0] // ranks)
    pad = tensor.new_zeros(rows * ranks - tensor.shape[0], *tensor.shape[1:])
    return list(torch.cat([tensor, pad]).split(rows))


def local_rows(rows, ranks, rank):
    """Real (unpadded) rows rank \`rank\` holds."""
    per = -(-rows // ranks)
    return max(0, min(rows, (rank + 1) * per) - rank * per)


# %% [unit] One unit: all-gather, compute, free
def all_gather(shards):
    """Every rank sends its chunk and receives everyone's: (N−1)/N of the tensor sent per rank."""
    return torch.cat(shards)


def unshard(unit_shards, shapes):
    """A unit's full tensors from the ranks' chunks (padding dropped), as every rank has them after the all-gather.
    FSDP frees them after the unit's forward and gathers them again for its backward."""
    return [all_gather(chunks)[: shape[0]].view(shape) for chunks, shape in zip(unit_shards, shapes)]


# %% [reduce-scatter] Reduce-scatter of a unit's gradients
def reduce_scatter(grads, ranks):
    """\`grads\` holds every rank's full gradient of one tensor. Each is padded and cut into the same row chunks; chunk
    r of all ranks is summed, divided by the number of ranks (the average) and kept by rank r only. Each rank sends
    (N−1)/N of the gradient."""
    chunks = [shard(g, ranks) for g in grads]
    return [torch.stack([chunks[q][r] for q in range(ranks)]).sum(0) / ranks for r in range(ranks)]


# %% [prefetch] Prefetching
def schedule(blocks, prefetch):
    """The order in which a rank issues all-gathers (AG), reduce-scatters (RS) and starts computing (F, B) in one
    step. FSDP2 prefetches in backward by default: entering a unit's backward, it first asks for the previous unit's
    weights. With \`prefetch\`, every unit's forward also asks for the next unit's weights before computing (and the
    all-gathers are asynchronous), so they arrive while it computes."""
    units = [f"h.{i}" for i in range(blocks)]
    out = [("AG", "root")]
    if prefetch:
        out.append(("AG", units[0]))
    out.append(("F", "embed"))
    for i, u in enumerate(units):
        if not prefetch:
            out.append(("AG", u))
        elif i + 1 < blocks:
            out.append(("AG", units[i + 1]))
        out.append(("F", u))
    out += [("F", "head"), ("AG", units[-1]), ("B", "head")]
    for i in reversed(range(blocks)):
        if i > 0:
            out.append(("AG", units[i - 1]))
        out += [("B", units[i]), ("RS", units[i])]
    return out + [("B", "embed"), ("RS", "root")]


# %% [memory] Memory per rank
def rank_bytes(shapes, cfg, rank):
    """A rank's weights and gradients: its chunk of every tensor, padding included (the chunks live in padded
    buffers); its AdamW state: two moments of its real rows."""
    b = cfg.bytes_per_value
    weights = optimizer = 0
    for shape in shapes:
        width = 1
        for d in shape[1:]:
            width *= d
        weights += -(-shape[0] // cfg.ranks) * width * b
        optimizer += 2 * local_rows(shape[0], cfg.ranks, rank) * width * b
    return {"weights": weights, "gradients": weights, "optimizer": optimizer}


# %% [step] One FSDP step
def fsdp_step(replicas, shards, optimizers, batch, loss_fn, cfg):
    """Rank r gathers the weights (here all at once; FSDP gathers each unit just before it computes, which changes
    the memory, not the arithmetic), runs its micro-batch forward and backward, reduce-scatters the gradients and
    updates its own chunks with its own AdamW. \`shards[r][i]\` is rank r's chunk of tensor i (an nn.Parameter).
    In float64 this step matches one process on the whole batch to float64 rounding (the check). In float32 it
    differs only by the order of the sums across ranks, which AdamW's division by √v̂ + ε magnifies where a
    gradient is near zero; the recorded run's differences are the size of one process's own when it adds the
    four micro-batches' gradients in turn instead of the whole batch at once."""
    ranks = cfg.ranks
    shapes = [p.shape for p in replicas[0].parameters()]
    full_grads = []
    for r, (model, x) in enumerate(zip(replicas, batch.chunk(ranks))):
        with torch.no_grad():
            for i, p in enumerate(model.parameters()):
                p.copy_(all_gather([shards[q][i] for q in range(ranks)])[: shapes[i][0]].view(shapes[i]))
        model.zero_grad(set_to_none=True)
        loss_fn(model, x).backward()
        full_grads.append([p.grad for p in model.parameters()])
    for i in range(len(shapes)):
        for r, g in enumerate(reduce_scatter([full_grads[q][i] for q in range(ranks)], ranks)):
            shards[r][i].grad = g
    for opt in optimizers:
        opt.step()
`;export{e as default};
