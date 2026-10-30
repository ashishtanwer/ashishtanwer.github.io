const e=`# One training step of GPT-2 small, the Training tab, written as plain PyTorch blocks.
# Each block has a forward pass and a hand-written backward pass: forward() keeps what backward()
# needs, and backward(∂L/∂output) fills the block's .grad fields and returns ∂L/∂input.
# Autograd is not used anywhere; the checks compare every gradient with autograd on transformers' GPT-2.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50257
    max_positions: int = 1024
    d_model: int = 768
    n_heads: int = 12  # 768 / 12 = 64 dimensions per head
    n_layers: int = 12
    d_mlp: int = 3072  # 4 × d_model
    eps: float = 1e-5  # dropout is off (p = 0), so the step is deterministic


def accumulate(parameter, grad):
    """Add to .grad: a weight used twice (the tied embedding) gets the sum of both gradients."""
    parameter.grad = grad if parameter.grad is None else parameter.grad + grad


# %% [layers] Linear and LayerNorm
class Linear(nn.Linear):
    """y = x Wᵀ + b. Backward: ∂L/∂W = Σ dyᵀ x, ∂L/∂b = Σ dy, ∂L/∂x = dy W."""

    def forward(self, x):
        self.x = x
        return super().forward(x)

    def backward(self, dy):
        accumulate(self.weight, dy.flatten(0, -2).T @ self.x.flatten(0, -2))  # summed over batch and positions
        if self.bias is not None:
            accumulate(self.bias, dy.flatten(0, -2).sum(dim=0))
        return dy @ self.weight


class LayerNorm(nn.LayerNorm):
    """x̂ = (x − mean) / std, y = γ x̂ + β, over each vector's 768 coordinates."""

    def forward(self, x):
        self.rstd = (x.var(dim=-1, unbiased=False, keepdim=True) + self.eps).rsqrt()
        self.x_hat = (x - x.mean(dim=-1, keepdim=True)) * self.rstd
        return self.x_hat * self.weight + self.bias

    def backward(self, dy):
        accumulate(self.weight, (dy * self.x_hat).flatten(0, -2).sum(dim=0))
        accumulate(self.bias, dy.flatten(0, -2).sum(dim=0))
        g = dy * self.weight  # ∂L/∂x̂
        # Every coordinate moves the mean and the variance, so g loses its mean and its x̂ component.
        return self.rstd * (g - g.mean(dim=-1, keepdim=True) - self.x_hat * (g * self.x_hat).mean(dim=-1, keepdim=True))


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token embedding plus a learned position embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.position = nn.Embedding(c.max_positions, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        self.ids = ids
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.token(ids) + self.position(positions)  # x₀: (batch, seq, d_model)

    def backward(self, dx):  # dx = ∂L/∂x₀
        # Both looked-up rows receive all of dx; a token id that occurs twice adds up its rows.
        d_token = torch.zeros_like(self.token.weight).index_add_(0, self.ids.flatten(), dx.reshape(-1, dx.shape[-1]))
        d_position = torch.zeros_like(self.position.weight)
        d_position[: dx.shape[1]] = dx.sum(dim=0)  # rows past the sequence get no gradient
        accumulate(self.token.weight, d_token)
        accumulate(self.position.weight, d_position)


# %% [attention] Multi-head self-attention
class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.qkv = Linear(c.d_model, 3 * c.d_model)  # Q, K and V for every head at once
        self.out = Linear(c.d_model, c.d_model)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # Split into heads: (batch, heads, seq, head_dim)
        self.q, self.k, self.v = (z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in (q, k, v))
        scores = self.q @ self.k.transpose(-2, -1) / math.sqrt(self.q.shape[-1])
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        self.weights = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        heads = self.weights @ self.v  # (batch, heads, seq, head_dim)
        return self.out(heads.transpose(1, 2).reshape(b, t, d))

    def backward(self, dy):  # dy = ∂L/∂output: (batch, seq, d_model)
        b, t, d = dy.shape
        d_heads = self.out.backward(dy).view(b, t, self.n_heads, -1).transpose(1, 2)
        d_weights = d_heads @ self.v.transpose(-2, -1)  # (batch, heads, seq, seq)
        d_v = self.weights.transpose(-2, -1) @ d_heads
        # Softmax: ∂s = p ⊙ (∂p − Σ p ∂p) along each row; masked entries have p = 0, so they get nothing.
        d_scores = self.weights * (d_weights - (self.weights * d_weights).sum(dim=-1, keepdim=True))
        d_scores = d_scores / math.sqrt(self.q.shape[-1])
        d_q, d_k = d_scores @ self.k, d_scores.transpose(-2, -1) @ self.q
        d_qkv = torch.cat([z.transpose(1, 2).reshape(b, t, d) for z in (d_q, d_k, d_v)], dim=-1)
        return self.qkv.backward(d_qkv)


# %% [mlp] MLP
def gelu(x):  # GPT-2's tanh approximation
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def gelu_grad(x):
    t = torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x**2)


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = Linear(c.d_model, c.d_mlp)
        self.down = Linear(c.d_mlp, c.d_model)

    def forward(self, x):
        self.h = self.up(x)  # (batch, seq, 3072)
        return self.down(gelu(self.h))

    def backward(self, dy):
        return self.up.backward(self.down.backward(dy) * gelu_grad(self.h))


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_1 = LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c)
        self.ln_2 = LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

    def backward(self, dy):
        # A residual sends the gradient both straight through and into the branch; the two add up.
        dx = dy + self.ln_2.backward(self.mlp.backward(dy))
        return dx + self.ln_1.backward(self.attention.backward(dx))


# %% [output] Probabilities
class Output(nn.Module):
    """Final LayerNorm, then a projection back to one logit per vocabulary token."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln_f = LayerNorm(c.d_model, eps=c.eps)
        self.lm_head = Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding

    def forward(self, x):
        return self.lm_head(self.ln_f(x))  # logits: (batch, seq, vocab)

    def backward(self, d_logits):
        return self.ln_f.backward(self.lm_head.backward(d_logits))


# %% [loss] Cross-entropy loss
def next_token_loss(logits, ids):
    """Mean of −log p(next token) over every position that has a next token, and ∂L/∂logits."""
    targets = ids[:, 1:]  # position t predicts token t + 1; the last position predicts nothing
    log_p = logits[:, :-1].log_softmax(dim=-1)
    loss = -log_p.gather(-1, targets[..., None]).mean()
    # ∂L/∂z = (p − onehot(target)) / N at the N predicting positions, and 0 at the last one.
    d_logits = torch.zeros_like(logits)
    d_logits[:, :-1] = log_p.exp() - F.one_hot(targets, logits.shape[-1])
    return loss, d_logits / targets.numel()


# %% [backward] Backward pass
def backward(model, d_logits):
    """Fill every parameter's .grad by passing ∂L/∂(each block's output) back through the blocks in reverse."""
    for parameter in model.parameters():
        parameter.grad = None
    dx = model.output.backward(d_logits)  # the LM head adds its gradient to the tied token embedding
    for block in reversed(model.blocks):
        dx = block.backward(dx)  # ∂L/∂(this block's input)
    model.embedding.backward(dx)  # … and the lookup adds its rows' gradient to the same matrix


# %% [adamw] AdamW update
class AdamW:
    """torch.optim.AdamW written out. The defaults are Hugging Face Trainer's lr and PyTorch's AdamW defaults."""

    def __init__(self, parameters, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.parameters = list(parameters)  # one group: biases, LayerNorm gains and embeddings included
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.parameters]  # running mean of g
        self.v = [torch.zeros_like(p) for p in self.parameters]  # running mean of g²
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p, m, v in zip(self.parameters, self.m, self.v):
            g = p.grad
            p.mul_(1 - self.lr * self.weight_decay)  # decoupled weight decay: shrink θ itself, not g
            m.mul_(b1).add_((1 - b1) * g)
            v.mul_(b2).add_((1 - b2) * g * g)
            m_hat = m / (1 - b1**self.t)  # bias correction: m and v start at 0
            v_hat = v / (1 - b2**self.t)
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))  # at step 1 this is ≈ −lr · sign(g)


# %% [model] Full model
class GPT2(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


@torch.no_grad()
def training_step(model, optimizer, ids):
    logits = model(ids)  # forward
    loss, d_logits = next_token_loss(logits, ids)
    backward(model, d_logits)  # backward
    optimizer.step()  # update
    return loss


# model = GPT2(); optimizer = AdamW(model.parameters())
# loss = training_step(model, optimizer, ids)  # ids: "To be, or not to be", 7 tokens → 6 targets
`;export{e as default};
