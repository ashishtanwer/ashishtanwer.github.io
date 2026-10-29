const e=`# SmolVLA (lerobot/smolvla_base, 450 M parameters; the tab records its LIBERO finetune lerobot/smolvla_libero),
# written as plain PyTorch blocks. SmolVLM2-500M, cut to its first 16 of 32 language layers, reads two camera
# images (64 tokens each after a pixel shuffle), the instruction and the robot state. An action expert 0.75 as
# wide, one layer beside each VLM layer, turns a chunk of 50 noisy actions into a velocity; 10 Euler steps of
# flow matching carry the chunk from noise (t = 1) to actions (t = 0). Expert layers alternate: even ones
# self-attend (to the prefix and to earlier actions), odd ones cross-attend (to the VLM layer's keys and values
# only). Each "# %% [id]" section is one step of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    image_size: int = 512  # LIBERO's 256 × 256 frames are resized to 512 (padding keeps the aspect)
    patch: int = 16
    vision_dim: int = 768  # SigLIP base
    vision_layers: int = 12
    vision_heads: int = 12
    vision_ffn: int = 3072
    shuffle: int = 4  # 32 × 32 patches → 8 × 8 tokens
    vocab: int = 49280
    d_model: int = 960  # SmolLM2's width
    vlm_layers: int = 16  # SmolVLM2 has 32; SmolVLA keeps the lower half
    n_heads: int = 15
    n_kv_heads: int = 5  # grouped-query attention: 3 query heads share each key/value head
    head_dim: int = 64
    d_ffn: int = 2560
    expert_width: int = 720  # 0.75 × 960
    expert_ffn: int = 2048
    self_attn_every: int = 2  # expert layers 0, 2, 4, … self-attend; 1, 3, 5, … cross-attend
    chunk: int = 50  # actions per chunk
    max_action_dim: int = 32  # LIBERO's 7 action and 8 state numbers are zero-padded to 32
    max_state_dim: int = 32
    steps: int = 10  # Euler steps of flow matching
    rope_base: float = 10000.0
    min_period: float = 4e-3  # timestep sinusoid
    max_period: float = 4.0
    eps: float = 1e-5  # RMSNorm
    vision_eps: float = 1e-6  # LayerNorm


# %% [vision] SigLIP image encoder
class VisionLayer(nn.Module):
    """A pre-norm ViT layer: LayerNorm, 12-head self-attention over all patches, LayerNorm, GELU MLP."""

    def __init__(self, c: Config):
        super().__init__()
        d = c.vision_dim
        self.heads = c.vision_heads
        self.norm1 = nn.LayerNorm(d, eps=c.vision_eps)
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))
        self.norm2 = nn.LayerNorm(d, eps=c.vision_eps)
        self.up = nn.Linear(d, c.vision_ffn)
        self.down = nn.Linear(c.vision_ffn, d)

    def forward(self, x):  # x: (batch, patches, 768)
        b, n, d = x.shape
        h = self.norm1(x)
        q, k, v = (proj(h).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        x = x + self.out((weights @ v).transpose(1, 2).reshape(b, n, d))
        return x + self.down(F.gelu(self.up(self.norm2(x)), approximate="tanh"))


class VisionEncoder(nn.Module):
    """16 × 16-pixel patches → 768-wide vectors, plus a learned position for each of the 32 × 32 places."""

    def __init__(self, c: Config):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, c.vision_dim, c.patch, stride=c.patch)
        self.position_embedding = nn.Embedding((c.image_size // c.patch) ** 2, c.vision_dim)
        self.layers = nn.ModuleList(VisionLayer(c) for _ in range(c.vision_layers))
        self.post_norm = nn.LayerNorm(c.vision_dim, eps=c.vision_eps)

    def forward(self, image):  # image: (batch, 3, 512, 512) in [-1, 1]
        x = self.patch_embedding(image).flatten(2).transpose(1, 2)  # (batch, 1024, 768)
        x = x + self.position_embedding.weight[None, : x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        return self.post_norm(x)


# %% [pixel-shuffle] Visual token reduction
def pixel_shuffle(x, s=4):
    """No weights: each s × s block of patch vectors is laid end to end as one vector, in raster order.
    (batch, 32·32, 768) → (batch, 8·8, 16·768): 16 × fewer tokens, 16 × wider."""
    b, n, d = x.shape
    side = int(n**0.5)
    x = x.view(b, side // s, s, side // s, s, d)  # (batch, block row, i, block col, j, d)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, (side // s) ** 2, s * s * d)


class Connector(nn.Module):
    """Pixel shuffle, then one linear map (no bias) from 12288 to the language model's 960."""

    def __init__(self, c: Config):
        super().__init__()
        self.shuffle = c.shuffle
        self.proj = nn.Linear(c.vision_dim * c.shuffle**2, c.d_model, bias=False)

    def forward(self, patches):
        return self.proj(pixel_shuffle(patches, self.shuffle))  # (batch, 64, 960)


# %% [prefix] Prefix: images, instruction and state
def embed_prefix(c: Config, image_tokens, lang_emb, lang_mask, state_emb):
    """The VLM's input: 64 tokens per camera, the instruction's tokens (padded to 48), then one state token.
    Image and word vectors are scaled by √960; the state is a linear map of the 32 padded state numbers.
    \`flags\` marks where a new attention block starts: 0 for every image and word token, 1 for the state."""
    scale = math.sqrt(c.d_model)
    parts = [tokens * scale for tokens in image_tokens] + [lang_emb * scale, state_emb[:, None]]
    b = state_emb.shape[0]
    pad = [torch.ones(b, t.shape[1], dtype=torch.bool) for t in image_tokens]
    pad += [lang_mask.bool(), torch.ones(b, 1, dtype=torch.bool)]
    flags = [0] * sum(t.shape[1] for t in image_tokens) + [0] * lang_emb.shape[1] + [1]
    return torch.cat(parts, dim=1), torch.cat(pad, dim=1), torch.tensor(flags)[None].expand(b, -1)


# %% [attention-mask] Attention mask
def attention_mask(pad, flags):
    """Token i may read token j when j is not padding and j's block started no later than i's:
    cumsum(flags)[j] ≤ cumsum(flags)[i]. Prefix flags 0…0 1 make images and words one bidirectional block
    that cannot see the state; action flags 1 1 … 1 make every action its own block, so the chunk is causal."""
    blocks = torch.cumsum(flags, dim=1)
    allowed = blocks[:, None, :] <= blocks[:, :, None]  # (batch, query, key)
    return allowed & pad[:, None, :] & pad[:, :, None]


def suffix_mask(prefix_pad, chunk):
    """What the 50 action tokens may read while denoising: every real prefix token, and the actions up to
    their own (the cumsum rule with flags 1 … 1). Cross-attention layers use only the first part."""
    b = prefix_pad.shape[0]
    causal = torch.ones(chunk, chunk, dtype=torch.bool).tril()
    return torch.cat([prefix_pad[:, None, :].expand(b, chunk, -1), causal[None].expand(b, -1, -1)], dim=2)


# %% [rope] Rotary positions
def rope(x, positions, base=10000.0):
    """Rotate each head's two halves by angle position / base^(2i/64). Positions count real tokens only
    (cumsum of the padding mask − 1), so padded words do not push the state and actions further away."""
    half = x.shape[-1] // 2
    frequencies = base ** (-torch.arange(half, dtype=torch.float32) * 2 / x.shape[-1])
    angles = positions[..., None].float() * frequencies  # (batch, tokens, 32)
    cos, sin = angles.cos()[:, :, None], angles.sin()[:, :, None]  # broadcast over heads
    x1, x2 = x.float().split(half, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def attend(q, k, v, mask, kv_groups):
    """Softmax attention with grouped keys: (batch, tokens, heads, 64) in, (batch, tokens, heads·64) out."""
    k, v = (t.repeat_interleave(kv_groups, dim=2) for t in (k, v))
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask[:, None], torch.finfo(scores.dtype).min)
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v.float())
    return out.flatten(2)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class SwiGLU(nn.Module):
    def __init__(self, d, ffn):
        super().__init__()
        self.gate = nn.Linear(d, ffn, bias=False)
        self.up = nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# %% [vlm-layer] A SmolVLM layer over the prefix
class VLMLayer(nn.Module):
    """RMSNorm → 15-head grouped-query self-attention over the 177 prefix tokens → RMSNorm → SwiGLU.
    It runs once per chunk; its keys and values (after RoPE) are kept for the expert's 10 steps."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.norm1 = RMSNorm(c.d_model, c.eps)
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)
        self.norm2 = RMSNorm(c.d_model, c.eps)
        self.mlp = SwiGLU(c.d_model, c.d_ffn)

    def forward(self, x, mask, positions):
        c = self.c
        b, n, _ = x.shape
        h = self.norm1(x)
        q = rope(self.q(h).view(b, n, -1, c.head_dim), positions, c.rope_base)
        k = rope(self.k(h).view(b, n, -1, c.head_dim), positions, c.rope_base)
        v = self.v(h).view(b, n, -1, c.head_dim)
        x = x + self.out(attend(q, k, v, mask, c.n_heads // c.n_kv_heads))
        return x + self.mlp(self.norm2(x)), (k, v)  # (k, v): this layer's cache


# %% [action-embedding] Noisy actions and time
def time_sinusoid(t, dim, min_period=4e-3, max_period=4.0):
    """t ∈ [0, 1] as sin and cos at periods from 0.004 to 4 (geometric): fine detail near the end of the flow."""
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    angles = (2 * math.pi / period)[None] * t[:, None].double()
    return torch.cat([angles.sin(), angles.cos()], dim=1).float()


class ActionTime(nn.Module):
    """Each of the 50 noisy actions → 720; the time sinusoid is concatenated to every one; an MLP fuses them."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.action_in = nn.Linear(c.max_action_dim, c.expert_width)
        self.mlp_in = nn.Linear(2 * c.expert_width, c.expert_width)
        self.mlp_out = nn.Linear(c.expert_width, c.expert_width)

    def forward(self, x_t, t):  # x_t: (batch, 50, 32), t: (batch,)
        c = self.c
        actions = self.action_in(x_t)
        time = time_sinusoid(t, c.expert_width, c.min_period, c.max_period)[:, None].expand_as(actions)
        return self.mlp_out(F.silu(self.mlp_in(torch.cat([actions, time], dim=-1))))  # (batch, 50, 720)


# %% [expert] Action expert: self- and cross-attention
class ExpertLayer(nn.Module):
    """One expert layer beside VLM layer i. Self-attention: the actions' own q, k, v (720 → 960 / 320 / 320),
    with the VLM's cached keys and values in front. Cross-attention: only the actions' queries; the VLM's
    cached keys and values (320 wide) go through this layer's own k and v maps, and the query positions
    restart at 0. Both use the VLM's 15 heads of 64."""

    def __init__(self, c: Config, cross: bool):
        super().__init__()
        self.c, self.cross = c, cross
        kv_in = c.n_kv_heads * c.head_dim if cross else c.expert_width
        self.norm1 = RMSNorm(c.expert_width, c.eps)
        self.q = nn.Linear(c.expert_width, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(kv_in, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(kv_in, c.n_kv_heads * c.head_dim, bias=False)
        self.out = nn.Linear(c.n_heads * c.head_dim, c.expert_width, bias=False)
        self.norm2 = RMSNorm(c.expert_width, c.eps)
        self.mlp = SwiGLU(c.expert_width, c.expert_ffn)

    def forward(self, x, cache, mask, positions):
        c = self.c
        b, n, _ = x.shape
        prefix_k, prefix_v = cache  # (batch, prefix, 5, 64), after RoPE
        h = self.norm1(x)
        q = self.q(h).view(b, n, -1, c.head_dim)
        if self.cross:
            q = rope(q, positions - positions.min(dim=1, keepdim=True).values, c.rope_base)
            k = self.k(prefix_k.flatten(2)).view(b, -1, c.n_kv_heads, c.head_dim)
            v = self.v(prefix_v.flatten(2)).view(b, -1, c.n_kv_heads, c.head_dim)
            mask = mask[:, :, : k.shape[1]]
        else:
            q = rope(q, positions, c.rope_base)
            k = torch.cat([prefix_k, rope(self.k(h).view(b, n, -1, c.head_dim), positions, c.rope_base)], dim=1)
            v = torch.cat([prefix_v, self.v(h).view(b, n, -1, c.head_dim)], dim=1)
        x = x + self.out(attend(q, k, v, mask, c.n_heads // c.n_kv_heads))
        return x + self.mlp(self.norm2(x))


# %% [flow-matching] Ten Euler steps
def integrate(velocity, noise, steps=10):
    """Flow matching runs from noise at t = 1 to actions at t = 0: x ← x + dt · v(x, t) with dt = −1/10.
    Training taught v ≈ noise − actions along straight lines x_t = t · noise + (1 − t) · actions."""
    dt = -1.0 / steps
    x = noise
    for step in range(steps):
        t = torch.full((noise.shape[0],), 1.0 + step * dt)
        x = x + dt * velocity(x, t)
    return x


# %% [async] Asynchronous inference
class ActionQueue:
    """LeRobot's async client: the robot keeps executing queued actions while the server computes the next
    chunk. It sends an observation once the queue is at most \`threshold\` of a chunk; the chunk comes back
    stamped from that observation's step, actions for steps already executed are dropped, and actions for
    steps still queued are blended with the old ones (weighted_average: 0.3 old + 0.7 new)."""

    def __init__(self, chunk=50, threshold=0.5, old=0.3):
        self.chunk, self.threshold, self.old = chunk, threshold, old
        self.queue = {}  # step → action
        self.last = -1  # last executed step

    def ready(self):
        return len(self.queue) / self.chunk <= self.threshold

    def receive(self, actions, start):  # actions: (chunk, dim) predicted from the observation at step \`start\`
        incoming = {start + i: a for i, a in enumerate(actions) if start + i > self.last}
        self.queue = {
            s: self.old * self.queue[s] + (1 - self.old) * a if s in self.queue else a for s, a in incoming.items()
        }

    def pop(self):
        self.last = min(self.queue)
        return self.queue.pop(self.last)


# %% [model] SmolVLA
class SmolVLA(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.vision = VisionEncoder(c)
        self.connector = Connector(c)
        self.embed_tokens = nn.Embedding(c.vocab, c.d_model)
        self.state_proj = nn.Linear(c.max_state_dim, c.d_model)
        self.vlm = nn.ModuleList(VLMLayer(c) for _ in range(c.vlm_layers))
        self.action_time = ActionTime(c)
        self.expert = nn.ModuleList(ExpertLayer(c, cross=i % c.self_attn_every != 0) for i in range(c.vlm_layers))
        self.expert_norm = RMSNorm(c.expert_width, c.eps)
        self.action_out = nn.Linear(c.expert_width, c.max_action_dim)

    def forward(self, images, lang_tokens, lang_mask, state, noise):
        """images: list of (batch, 3, 512, 512) in [-1, 1]; state: (batch, 32) normalised and padded;
        noise: (batch, 50, 32). Returns the normalised action chunk (batch, 50, 32)."""
        c = self.c
        image_tokens = [self.connector(self.vision(image)) for image in images]
        x, pad, flags = embed_prefix(c, image_tokens, self.embed_tokens(lang_tokens), lang_mask, self.state_proj(state))
        mask, positions = attention_mask(pad, flags), torch.cumsum(pad, dim=1) - 1
        caches = []
        for layer in self.vlm:  # once per chunk
            x, cache = layer(x, mask, positions)
            caches.append(cache)
        action_mask = suffix_mask(pad, c.chunk)
        action_positions = pad.sum(dim=1, keepdim=True) + torch.arange(c.chunk)[None]

        def velocity(x_t, t):  # ten times per chunk
            h = self.action_time(x_t, t)
            for layer, cache in zip(self.expert, caches):
                h = layer(h, cache, action_mask, action_positions)
            return self.action_out(self.expert_norm(h))

        return integrate(velocity, noise, c.steps)
`;export{e as default};
