const e=`# LLaDA-8B-Instruct, the model in the Diffusion LM tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram. LLaDA is a Llama-style transformer without
# the causal mask: it predicts every [MASK] slot at once, and a sampler unmasks the answer over steps.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 126464
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # 128 dimensions per head; every head has its own K and V
    d_mlp: int = 12288
    rope_theta: float = 500_000.0
    eps: float = 1e-5
    mask_id: int = 126336  # the [MASK] token


# %% [rmsnorm] RMSNorm
class RMSNorm(nn.Module):
    """LLaDA computes every norm in float32, even when the model runs in bfloat16."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


# %% [embedding] Embedding
class Embedding(nn.Module):
    """Token lookup. [MASK] is an ordinary row of the table; positions enter through RoPE."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)

    def forward(self, ids):  # ids: (batch, seq)
        return self.token(ids)  # (batch, seq, d_model)


# %% [rope] RoPE
def rope(x, theta):
    """Rotate each pair (i, i + d/2) of every head vector by position × theta^(-2i/d)."""
    seq, d = x.shape[-2], x.shape[-1]
    frequencies = theta ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    angles = torch.arange(seq, device=x.device)[:, None] * frequencies  # (seq, d/2)
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# %% [attention] Bidirectional self-attention
class Attention(nn.Module):
    """Multi-head attention with no causal mask: every position, [MASK] included, attends to all."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.q = nn.Linear(c.d_model, c.d_model, bias=False)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x):  # x: (batch, seq, d_model)
        b, t, d = x.shape
        heads = lambda z: z.view(b, t, self.c.n_heads, -1).transpose(1, 2)  # (batch, heads, seq, 128)
        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        q, k = rope(q, self.c.rope_theta), rope(k, self.c.rope_theta)
        weights = (q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5).softmax(dim=-1)  # (batch, heads, seq, seq)
        return self.out((weights @ v).transpose(1, 2).reshape(b, t, d))


# %% [mlp] SwiGLU MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.up = nn.Linear(c.d_model, c.d_mlp, bias=False)
        self.down = nn.Linear(c.d_mlp, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm residual block: x + Attention(RMSNorm(x)), then x + MLP(RMSNorm(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.attention = Attention(c)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final RMSNorm, then a projection to one logit per vocabulary token (not tied to the embedding)."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = RMSNorm(c.d_model, c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

    def forward(self, x):
        return self.lm_head(self.norm(x))  # logits: (batch, seq, vocab)


def predictions(logits):
    """Every slot's softmax, its most likely token and that token's probability (its confidence)."""
    probabilities = logits.softmax(dim=-1)
    confidence, tokens = probabilities.max(dim=-1)
    return probabilities, tokens, confidence


# %% [masking] Masked diffusion objective
def forward_process(ids, prompt_length, mask_id, eps=1e-3):
    """Draw a mask ratio t per sequence and replace each answer token by [MASK] with probability t.
    The prompt is never masked. ids: (batch, seq); prompt_length: (batch,)."""
    b, l = ids.shape
    t = torch.rand(b, device=ids.device)
    p_mask = ((1 - eps) * t + eps)[:, None].expand(b, l)  # t in (eps, 1]
    masked = torch.rand(b, l, device=ids.device) < p_mask
    masked &= torch.arange(l, device=ids.device) >= prompt_length[:, None]
    return torch.where(masked, mask_id, ids), masked, p_mask


def masked_diffusion_loss(model, ids, prompt_length, mask_id):
    """Cross-entropy at the masked slots only, weighted by 1 / t and averaged over answer tokens."""
    noisy, masked, p_mask = forward_process(ids, prompt_length, mask_id)
    logits = model(noisy)
    answer_length = (ids.shape[1] - prompt_length)[:, None].expand_as(ids)
    loss = F.cross_entropy(logits[masked], ids[masked], reduction="none") / p_mask[masked]
    return (loss / answer_length[masked]).sum() / ids.shape[0]


# %% [remasking] Low-confidence remasking
def transfer_schedule(n_masked, steps):
    """How many slots each step unmasks: an equal share, the remainder going to the first steps."""
    base, extra = divmod(n_masked, steps)
    return [base + (1 if step < extra else 0) for step in range(steps)]


@torch.no_grad()
def denoise_step(model, x, n_unmask, mask_id):
    """Predict every slot at once, unmask the n_unmask most confident [MASK] slots and remask the
    rest: they stay [MASK] and are predicted again next step. x: (batch, seq)."""
    masked = x == mask_id
    _, tokens, confidence = predictions(model(x))
    confidence = confidence.masked_fill(~masked, float("-inf"))  # only [MASK] slots compete
    chosen = confidence.topk(n_unmask, dim=-1).indices  # (batch, n_unmask)
    return x.scatter(1, chosen, tokens.gather(1, chosen))


@torch.no_grad()
def generate(model, prompt, answer_length, steps, mask_id):
    """Start the answer as all [MASK] and denoise it in \`steps\` steps (temperature 0, one block)."""
    x = F.pad(prompt, (0, answer_length), value=mask_id)
    for n_unmask in transfer_schedule(answer_length, steps):
        x = denoise_step(model, x, n_unmask, mask_id)
    return x


# %% [model] Full model
class LLaDA(nn.Module):
    """The mask predictor: a bidirectional transformer from every slot to vocabulary logits."""

    def __init__(self, c: Config = Config()):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.output = Output(c)

    def forward(self, ids):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


# answer = generate(LLaDA(), prompt_ids, answer_length=24, steps=24, mask_id=Config.mask_id)
`;export{e as default};
