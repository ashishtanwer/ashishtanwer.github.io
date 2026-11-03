const e=`# Inductor tiling GPT-2 small's MLP up-projection on the CPU, as the Compilers Explainer records it: the GEMM and the
# GELU after it, compiled with weights frozen and max-autotune so Inductor writes its own C++ GEMM template; the
# loop nest's tile sizes read back out of that code; Inductor's rule for the cache blocks, from the CPU's L1 and L2;
# and the knobs that override it. The check compiles again and compares outputs and tile sizes with the recording.
import copy
import inspect
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn
from torch._inductor import config as inductor


@dataclass
class Config:
    d_model: int = 768
    d_ff: int = 3072


# %% [gemm] The GEMM: GPT-2's MLP up-projection
class Up(nn.Module):
    """c_fc and the GELU after it: (T, 768) @ (768, 3072) + bias, then GELU, the largest GEMM of the block."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.c_fc = nn.Linear(c.d_model, c.d_ff)

    def forward(self, x):
        return F.gelu(self.c_fc(x), approximate='tanh')


def from_gpt2(state, layer=0):
    up = Up()
    up.c_fc.weight.data = state[f'h.{layer}.mlp.c_fc.weight'].t().contiguous()  # Conv1D (in, out) → Linear (out, in)
    up.c_fc.bias.data = state[f'h.{layer}.mlp.c_fc.bias'].clone()
    return up.eval()


# %% [compile] Compiling it to Inductor's C++ GEMM
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


def compile_gemm(module, x, cache_blocking=None, register_blocking=None, thread_factors=None):
    """Freeze the weights (so they become constants Inductor may repack) and turn on max-autotune with only the C++
    template allowed, so the GEMM is Inductor's own code. cache_blocking 'Mc,Nc,Kc' and thread_factors 'Mt,Nt,Kt'
    override Inductor's choices; register_blocking (Mr, Nr, Kr) keeps only that micro-kernel. Inductor's FX-graph
    cache is off, since its key does not see the micro-kernel choice. Returns the compiled copy of the module, its
    output and the generated code."""
    from torch._inductor import select_algorithm  # noqa: F401 — cpp_micro_gemm imported before it is circular
    from torch._inductor.codegen import cpp_micro_gemm
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


# %% [blocking] The tile sizes in the code
BLOCKING = ('num_threads', 'M', 'N', 'K', 'Mr', 'Nr', 'Kr', 'Mt_blocks', 'Nt_blocks', 'Kt_blocks',
            'Mc_blocks', 'Nc_blocks', 'Kc_blocks')


def blocking(code):
    """The constexprs at the top of the GEMM kernel: register tile Mr × Nr (× Kr), each thread's share in register
    tiles (Mt, Nt, Kt blocks) and the cache block (Mc, Nc, Kc blocks), as Inductor wrote them."""
    found = {}
    for name in BLOCKING:
        m = re.search(rf'constexpr int64_t {name} = (?:static_cast<int64_t>\\()?(\\d+)', code)
        if m:
            found[name] = int(m[1])
    return found


# %% [heuristic] Inductor's rule for the cache blocks
def cache_blocking(b, l1_bytes, l2_bytes, itemsize=4):
    """What Inductor's C++ GEMM template picks when nothing overrides it (cpp_gemm_template.py, 'CPP GEMM Cache
    Blocking Algorithm'): a Kc × Nr panel of B must fit in 80 % of L1, so Kc = ⌊0.8·L1 / (Kr·Nr·4 bytes)⌋ unless
    all of K fits; then, if an Mr × Kt strip of A fits in half of L2, take as many rows of A as fit (Mc) and one
    register block of B at a time (Nc = 1); otherwise keep A's rows and C's columns square inside half of L2."""
    L1, L2 = 0.8 * l1_bytes, 0.5 * l2_bytes
    Kc = b['Kt_blocks']
    if b['Kr'] * b['Kt_blocks'] * b['Nr'] * itemsize > L1:
        Kc = math.floor(L1 / (b['Kr'] * b['Nr'] * itemsize))
    min_Mc = math.ceil(2 * b['Mr'] / b['Nr'])
    Kt_bytes = b['Kt_blocks'] * b['Kr'] * itemsize
    if min_Mc * b['Mr'] * Kt_bytes < L2:
        Mc = min(b['Mt_blocks'], math.floor(L2 / (b['Mr'] * Kt_bytes)))
        Nc = 1
    else:
        Mc = b['Mt_blocks']
        Nc = min(math.ceil(Mc * b['Mr'] / b['Nr']), b['Nt_blocks'])
        Kc_bytes = Kc * b['Kr'] * itemsize
        if Mc * b['Mr'] * (Kc_bytes + Nc * b['Nr'] * 4) > L2:
            M_max = (math.sqrt(Kc_bytes * Kc_bytes + 16 * L2) - Kc_bytes) / 8
            if M_max < Mc * b['Mr']:
                Mc = math.floor(M_max / b['Mr'])
                Nc = min(math.ceil(Mc * b['Mr'] / b['Nr']), b['Nt_blocks'])
    return Mc, Nc, Kc
`;export{e as default};
