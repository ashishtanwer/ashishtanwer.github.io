const n=`# TorchDynamo capturing GPT-2 small's first block, as the Compilers Explainer records it: the block in plain PyTorch
# with GPT-2's real layer-0 weights, a backend that keeps the FX graph Dynamo hands it, a debugging version of the
# block whose print, .item() and if-on-a-tensor break the graph, the guards that decide when a compiled graph may be
# reused, and the calls that make Dynamo compile again. The check compares these with the recorded capture.
import math
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


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


# %% [capture] Capturing the FX graph
def capture(module, *inputs):
    """torch.compile with a backend that keeps every FX graph Dynamo hands it and runs it unchanged (no Inductor):
    the graph is exactly what Dynamo traced from the bytecode."""
    graphs = []

    def keep(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    with torch.no_grad():
        out = torch.compile(module, backend=keep)(*inputs)
    return out, graphs


def node_counts(gm):
    """How many nodes of each kind: placeholders (inputs and lifted parameters), calls, the output."""
    kinds = Counter(node.op for node in gm.graph.nodes)
    return [kinds[op] for op in ('placeholder', 'call_function', 'call_method', 'call_module', 'get_attr', 'output')]


# %% [graph-breaks] A debugging version that breaks the graph
class DebugBlock(Block):
    """The same block with three lines someone might add while debugging. Dynamo cannot put any of them in a graph:
    print is a side effect, .item() makes a Python number out of a tensor, and an if on a tensor's value needs the
    value. Each one ends the graph so far; Python runs the line, and Dynamo starts a new graph after it."""

    def forward(self, x):
        h = self.ln_1(x)
        print('attention input', tuple(h.shape))
        a = self.attn(h)
        peak = a.abs().max().item()
        if peak > 100:
            a = a.clamp(-100, 100)
        x = x + a
        h = self.mlp(self.ln_2(x))
        if h.norm() > x.norm():
            h = h / 2
        return x + h


def explain(module, x):
    """torch._dynamo.explain: the graphs, the breaks between them and why, and the ops in each graph."""
    torch._dynamo.reset()
    with torch.no_grad():
        report = torch._dynamo.explain(module)(x)
    return report


# %% [guards] The guards
def guard_kinds(module, x):
    """The guards of the clean capture, counted by kind: TENSOR_MATCH (dtype, device, shape, strides of a tensor),
    ID_MATCH and TYPE_MATCH (the same object or class as when traced), and the rest."""
    report = explain(module, x)
    return Counter(guard.create_fn_name() for guard in report.out_guards)


# %% [recompile] Compiling again when a guard fails
def compiles_per_call(block, calls):
    """Runs the compiled block on each (input, grad enabled) in turn, with the block cast to the input's dtype, and
    counts the graphs Dynamo compiles for each call: 0 means a cached graph's guards all passed."""
    compiled = []

    def keep(gm, example_inputs):
        compiled.append(gm)
        return gm.forward

    torch._dynamo.reset()
    fn = torch.compile(block, backend=keep)
    counts = []
    for x, grad in calls:
        before = len(compiled)
        block.to(x.dtype)
        with torch.set_grad_enabled(grad):
            fn(x)
        counts.append(len(compiled) - before)
    return counts, compiled
`;export{n as default};
