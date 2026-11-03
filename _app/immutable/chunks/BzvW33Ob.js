const e=`# Inductor fusing GPT-2 small's first block, as the Compilers Explainer records it: the block in plain PyTorch, the
# ATen graph Inductor receives after decomposition, the kernels it generates (C++ on the CPU, Metal on the GPU) read
# out of the code it writes, and the bytes each kernel reads and writes, taken from the buffer shapes in that code.
# The check compiles the block again and compares kernels and bytes with the recorded ones.
import math
import re
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch._inductor.utils import run_and_get_code


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


# %% [aten-graph] The ATen graph Inductor fuses
def aten_graph(block, x):
    """Compile with a post-grad pass that only looks: after Dynamo's capture and AOTAutograd's decomposition, every
    call in the graph is one ATen op (layer_norm is now mean, var, sub, mul…). Returns (op name, output shape) per
    call, in order."""
    ops = []

    def look(graph):
        for node in graph.nodes:
            if node.op == 'call_function':
                value = node.meta.get('val')
                target = node.target  # an ATen overload (aten.mul.Tensor → mul.Tensor), or a Python function (getitem)
                name = str(target).replace('aten.', '') if hasattr(target, '_schema') else target.__name__
                ops.append((name, tuple(value.shape) if hasattr(value, 'shape') else ()))

    torch._dynamo.reset()
    with torch._inductor.config.patch(post_grad_custom_post_pass=look), torch.no_grad():
        torch.compile(block)(x)
    return ops


# %% [kernels] The kernels Inductor writes
KERNEL_CALL = re.compile(r'^\\s*((?:cpp_fused|generated_kernel|cpp_)\\w*|extern_kernels\\.\\w+|torch\\.ops\\.\\w+\\.\\w+(?:\\.\\w+)?)\\((.*)\\)\\s*$')


def generated_code(block, x):
    """Compile with Inductor and return the output (run once) and the Python wrapper it wrote, which holds the C++ or
    Metal source of every kernel and, in call(), the order they run in."""
    with torch.no_grad():
        out, codes = run_and_get_code(torch.compile(block), x)
    return out, codes[-1]


def split_args(text):
    depth, part, parts = 0, '', []
    for ch in text:
        if ch == ',' and depth == 0:
            parts.append(part.strip())
            part = ''
            continue
        depth += ch in '([{'
        depth -= ch in ')]}'
        part += ch
    if part.strip():
        parts.append(part.strip())
    return parts


def kernels(code):
    """Every kernel launch in call(), in order: its name and its tensor arguments, each as (argument, buffer, numel,
    role), role 'r' (read), 'w' (written) or 'rw'. Generated kernels say which arguments they write in their signature
    (C++: const float* in_ptr / float* out_ptr; Metal: constant / device); library calls (extern_kernels.mm…) read
    their positional arguments and write out=."""
    signature = {}
    for name, types in re.findall(r"^(\\w+) = async_compile\\.cpp_pybinding\\((\\[.*?\\])", code, re.M):
        params = re.search(rf'void\\s+kernel\\(([^)]*)\\)', code[code.index(f'{name} = async_compile'):])
        signature[name] = ['rw' if 'in_out_ptr' in p else 'r' if 'const' in p else 'w'
                           for p in params[1].split(',')] if params else []
    for name in re.findall(r"^(\\w+) = async_compile\\.metal\\('(\\w+)'", code, re.M):
        name = name[0] if isinstance(name, tuple) else name
        body = code[code.index(f'kernel void {name}('):]
        params = body[body.index('(') + 1: body.index(')')].split(',')
        signature[name] = ['r' if 'constant' in p else 'w' for p in params if '_ptr' in p]
    sizes = {}
    body = code[code.index('def call(self, args):'):]
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r'assert_size_stride\\((\\w+), \\(([\\d, ]*)\\)', line)
        if m:
            sizes[m[1]] = math.prod(int(v) for v in m[2].replace(' ', '').split(',') if v)
        m = re.match(r"assert_size_stride_grouped\\(\\(([\\w, ]+)\\), \\((.*?)\\), \\(\\(", line)
        if m:
            names = [n.strip() for n in m[1].split(',') if n.strip()]
            shapes = re.findall(r'\\(([\\d, ]*)\\)', m[2])
            for n, s in zip(names, shapes):
                sizes[n] = math.prod(int(v) for v in s.replace(' ', '').split(',') if v)
        m = re.match(r'(\\w+) = empty_strided(?:_cpu|_mps)?\\(\\(([\\d, ]*)\\)', line)
        if m:
            sizes[m[1]] = math.prod(int(v) for v in m[2].replace(' ', '').split(',') if v)
        m = re.match(r'(\\w+) = (\\w+); del \\2\\s+# reuse', line)
        if m:
            sizes[m[1]] = sizes[m[2]]
        m = re.match(r'(\\w+) = reinterpret_tensor\\((\\w+), \\(([\\d, ]*)\\).*# reuse', line)
        if m:  # a dead buffer's memory, reused under a new shape
            sizes[m[1]] = math.prod(int(v) for v in m[3].replace(' ', '').split(',') if v)
    launches = []
    for line in body.splitlines():
        m = KERNEL_CALL.match(line)
        if not m:
            continue
        name, args = m[1], split_args(m[2])
        roles = signature.get(name)
        tensors = []
        for i, arg in enumerate(a for a in args if not a.startswith(('threads=', 'group_size='))):
            key, _, value = arg.partition('=') if re.match(r'^\\w+=', arg) else ('', '', arg)
            view = re.match(r'reinterpret_tensor\\((\\w+), \\(([\\d, ]*)\\)', value)
            buffer = view[1] if view else value
            if buffer not in sizes:
                continue
            numel = math.prod(int(v) for v in view[2].replace(' ', '').split(',') if v) if view else sizes[buffer]
            role = roles[i] if roles else ('w' if key == 'out' else 'r')
            if key and key not in ('out',):
                role = 'r'
            tensors.append((value, buffer, numel, role))
        launches.append((name, tensors))
    return launches


def summary(launches):
    """How many launches of each kind: generated (fused) kernels and library calls."""
    kinds = Counter('library' if name.startswith(('extern_kernels', 'torch.ops')) else 'generated'
                    for name, _ in launches)
    return kinds['generated'], kinds['library']


# %% [bytes] The bytes each kernel moves
def kernel_bytes(launch, itemsize=4):
    """Bytes a launch reads and writes: each distinct buffer (or view of one) once, as if every element were
    loaded or stored exactly once."""
    name, tensors = launch
    distinct = {(arg, role): numel for arg, _, numel, role in tensors}
    read = sum(numel for (_, role), numel in distinct.items() if 'r' in role)
    written = sum(numel for (_, role), numel in distinct.items() if 'w' in role)
    return read * itemsize, written * itemsize
`;export{e as default};
