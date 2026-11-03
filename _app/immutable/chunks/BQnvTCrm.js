const e=`# Inductor's autotuner on GPT-2 small's first block, as the Compilers Explainer records it on the CPU: with max-autotune
# every matrix multiply gets candidate kernels (ATen's library call and Inductor's own C++ GEMM template), each is timed
# on the real shapes and the fastest is kept. Then a wider search over the template's own knobs (register tile, cache
# block, thread split), which Inductor sets by rule rather than by timing. The check re-runs both and compares the
# candidates, the decisive winners and the tile sizes with the recording.
import copy
import inspect
import math
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn
from torch._inductor import config as inductor
from torch._inductor import select_algorithm


@dataclass
class Config:
    d_model: int = 768
    heads: int = 12
    eps: float = 1e-5


# %% [block] GPT-2's block in plain PyTorch
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.c_attn = nn.Linear(c.d_model, 3 * c.d_model)
        self.c_proj = nn.Linear(c.d_model, c.d_model)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c_fc = nn.Linear(c.d_model, 4 * c.d_model)
        self.c_proj = nn.Linear(4 * c.d_model, c.d_model)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate='tanh'))


class Block(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attn = Attention(c)
        self.ln_2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def from_gpt2(state, layer=0, block=None):
    """GPT-2's checkpoint stores its projections as Conv1D, (in, out); nn.Linear wants (out, in)."""
    block = block or Block()
    prefix = f'h.{layer}.'
    weights = {}
    for key, value in state.items():
        name = key[len(prefix):]
        if not key.startswith(prefix) or name in ('attn.bias', 'attn.masked_bias'):
            continue  # the causal-mask buffers, rebuilt in forward
        weights[name] = value.t().contiguous() if value.ndim == 2 else value
    block.load_state_dict(weights, strict=True)
    return block.eval()


# %% [autotune] Inductor's autotuner, candidate by candidate
def autotune(block, x):
    """Compile with frozen weights and max-autotune. For every GEMM Inductor lowers, its AlgorithmSelectorCache times
    each candidate on the real shapes and keeps the fastest; a feedback saver hears each verdict. Returns the output,
    per GEMM (op, input shapes, [(candidate, ms)…]) in the order Inductor tuned them, and the code it wrote."""
    from torch._inductor.graph import GraphLowering

    heard, codes = [], []

    def hear(timings, name, input_nodes, choices, profiled, precompile_times):
        shapes = [[int(s) for s in node.get_size()] for node in input_nodes]
        heard.append((name, shapes, sorted(((c.name, ms) for c, ms in timings.items()), key=lambda t: t[1])))

    select_algorithm.add_feedback_saver(hear)
    try:
        torch._dynamo.reset()
        with (inductor.patch({'freezing': True, 'max_autotune': True}),
              mock.patch.object(GraphLowering, 'save_output_code', codes.append), torch.no_grad()):
            out = torch.compile(block)(x)
    finally:
        select_algorithm.clear_feedback_savers()
    return out, heard, codes[-1]


def template_kernels(code):
    """How many of Inductor's own GEMM kernels the code defines: kernels built around its micro-GEMM (named after the
    ops fused into them, cpp_fused_addmm_…); the other GEMMs are library calls (extern_kernels.addmm…)."""
    sources = code.split('= async_compile.cpp_pybinding(')[1:]
    return sum('micro_gemm' in source.split("''')")[0] for source in sources)


# %% [search] A wider search over the template's knobs
def separate(module):
    """A shallow copy (same weights) whose forward is a new code object. Dynamo caches compiled graphs per code
    object, so each copy keeps its own compiled kernel and several variants can be timed side by side."""
    forward = type(module).forward
    scope = {}
    exec(compile(textwrap.dedent(inspect.getsource(forward)), f'<{type(module).__name__} copy>', 'exec'),
         sys.modules[forward.__module__].__dict__, scope)
    copy_ = copy.copy(module)
    copy_.__class__ = type(type(module).__name__, (type(module),), {'forward': scope['forward']})
    return copy_


def compile_gemm(module, x, register_blocking=None, cache_blocking=None, thread_factors=None):
    """One candidate: the module compiled to Inductor's C++ GEMM template only, with the register tile (Mr, Nr, Kr),
    the cache block 'Mc,Nc,Kc' and the thread split 'Mt,Nt,Kt' fixed instead of chosen by Inductor's rules.
    Inductor's FX-graph cache is off, since its key does not see the micro-kernel choice. Returns the compiled copy
    of the module, its output and the code."""
    from torch._inductor.codegen import cpp_micro_gemm  # after select_algorithm (top): the other order is circular
    from torch._inductor.graph import GraphLowering

    keep = cpp_micro_gemm.micro_gemm_configs
    if register_blocking is not None:
        cpp_micro_gemm.micro_gemm_configs = {
            cls: [c for c in configs if tuple(c.register_blocking) == tuple(register_blocking)]
            for cls, configs in keep.items()}
    codes = []
    try:
        with (inductor.patch({'freezing': True, 'max_autotune': True, 'max_autotune_gemm_backends': 'CPP',
                              'fx_graph_cache': False,
                              'cpp.gemm_cache_blocking': cache_blocking, 'cpp.gemm_thread_factors': thread_factors}),
              mock.patch.object(GraphLowering, 'save_output_code', codes.append), torch.no_grad()):
            compiled = torch.compile(separate(module))
            out = compiled(x)
    finally:
        cpp_micro_gemm.micro_gemm_configs = keep
    return compiled, out, codes[-1]


def tile_sizes(code):
    """(Mr, Nr, Mc, Nc, Kc, Mt, Nt, Kt) as written in the generated kernel's constexprs."""
    names = ('Mr', 'Nr', 'Mc_blocks', 'Nc_blocks', 'Kc_blocks', 'Mt_blocks', 'Nt_blocks', 'Kt_blocks')
    return tuple(int(re.search(rf'constexpr int64_t {n} = (\\d+)', code)[1]) for n in names)


def search(module, x, candidates, rounds, time_once, others=None):
    """Compile every candidate, then time them all in interleaved rounds: each round visits every candidate (and
    each reference in \`others\`) once, in a shuffled order, so a busy spell on the machine hits all of them alike.
    Keep the fastest by its best round, as an autotuner over these knobs would. Returns [(candidate, tile sizes,
    ms per round)] fastest first, and the references' ms per round."""
    compiled = [compile_gemm(module, x, **c) for c in candidates]
    fns = [(i, c[0]) for i, c in enumerate(compiled)] + list((others or {}).items())
    times = {key: [] for key, _ in fns}
    rng = random.Random(0)
    with torch.no_grad():
        for _ in range(rounds):
            rng.shuffle(fns)
            for key, fn in fns:
                times[key].append(time_once(lambda: fn(x)))
    results = [(c, tile_sizes(code), times[i]) for i, (c, (_, _, code)) in enumerate(zip(candidates, compiled))]
    return sorted(results, key=lambda r: min(r[2])), {k: times[k] for k in (others or {})}
`;export{e as default};
