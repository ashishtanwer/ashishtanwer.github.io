const e=`# NVIDIA Isaac GR00T N1.5 (3B), the vision-language-action model in the Embodied Explainer's GR00T tab, written
# as plain PyTorch blocks. System 2 is Eagle 2.5, frozen: a SigLIP vision tower reads the 224 × 224 camera frame
# as 256 patch tokens, a Linear projects them into Qwen3-1.7B, which reads them with the instruction; GR00T keeps
# only its first 12 of 28 layers and takes their (final-normed) output. System 1 is a 16-block diffusion
# transformer (DiT): the robot's state token, 32 learned future tokens and 16 noisy action tokens attend to the
# VLM tokens (even blocks) and to each other (odd blocks), with the flow time t entering every block through
# adaptive LayerNorm. Four Euler steps of flow matching turn noise into a 16-step action chunk. The state and
# action encoders and the action decoder keep one weight set per embodiment (32 slots; the GR1 humanoid is 24).
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    # System 2 · Eagle 2.5: SigLIP vision tower
    image_size: int = 224
    patch: int = 14  # 16 × 16 = 256 patch tokens
    vit_dim: int = 1152
    vit_heads: int = 16  # 72 dimensions per head
    vit_mlp: int = 4304
    vit_layers: int = 27
    # System 2 · Eagle 2.5: Qwen3-1.7B, cut after layer 12
    vocab: int = 151680
    llm_dim: int = 2048
    llm_heads: int = 16
    llm_kv_heads: int = 8  # two query heads share each key-value head
    llm_head_dim: int = 128
    llm_mlp: int = 6144
    llm_layers: int = 12  # of Qwen3-1.7B's 28; GR00T reads the output of layer 12
    rope_theta: float = 1_000_000.0
    image_token: int = 151669  # <IMG_CONTEXT>, replaced by the projected patch tokens
    # The VL adapter in the action head: LayerNorm + 4 self-attention blocks over the VLM tokens
    adapter_heads: int = 32  # 64 dimensions per head
    adapter_layers: int = 4
    # System 1 · the action head
    embodiments: int = 32  # weight slots of the state/action encoders and the decoder
    max_state: int = 64  # state vectors are zero-padded to 64
    max_action: int = 32  # action vectors are zero-padded to 32
    horizon: int = 16  # actions per chunk
    future_tokens: int = 32
    d_model: int = 1536  # DiT width, 32 heads × 48
    dit_heads: int = 32
    dit_layers: int = 16
    encoder_hidden: int = 1024  # the state encoder's and the decoder's hidden width; also the DiT's output
    max_positions: int = 1024  # learned positions added to the action tokens
    frequency_dim: int = 256  # the DiT's timestep sinusoid
    timestep_buckets: int = 1000  # t ∈ [0, 1) enters as int(1000 · t)
    steps: int = 4  # Euler steps of the flow


# %% [vision] SigLIP vision tower
class VisionLayer(nn.Module):
    """A pre-norm ViT layer: LayerNorm → 16-head attention → add; LayerNorm → GELU(tanh) MLP → add."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.vit_heads
        self.norm1 = nn.LayerNorm(c.vit_dim, eps=1e-6)
        self.q, self.k, self.v, self.out = (nn.Linear(c.vit_dim, c.vit_dim) for _ in range(4))
        self.norm2 = nn.LayerNorm(c.vit_dim, eps=1e-6)
        self.fc1 = nn.Linear(c.vit_dim, c.vit_mlp)
        self.fc2 = nn.Linear(c.vit_mlp, c.vit_dim)

    def forward(self, x):  # (batch, 256, 1152)
        h = self.norm1(x)
        q, k, v = (p(h).unflatten(-1, (self.heads, -1)).transpose(1, 2) for p in (self.q, self.k, self.v))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)  # every patch reads every patch
        x = x + self.out((weights @ v).transpose(1, 2).flatten(2))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class VisionTower(nn.Module):
    """224 × 224 pixels (scaled to [-1, 1]) → 256 patch vectors: a 14 × 14 stride-14 convolution, a learned
    position per patch, 27 layers and a final LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        side = c.image_size // c.patch
        self.patch = nn.Conv2d(3, c.vit_dim, kernel_size=c.patch, stride=c.patch)
        self.position = nn.Embedding(side * side, c.vit_dim)
        self.layers = nn.ModuleList(VisionLayer(c) for _ in range(c.vit_layers))
        self.norm = nn.LayerNorm(c.vit_dim, eps=1e-6)

    def forward(self, pixels):  # (images, 3, 224, 224)
        x = self.patch(pixels).flatten(2).transpose(1, 2) + self.position.weight  # (images, 256, 1152), row by row
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# %% [mlp-connector] Patch tokens into the language model
class Connector(nn.Module):
    """One Linear lifts each patch vector to the LLM's width; the 256 results replace the 256 <IMG_CONTEXT>
    placeholders in the embedded prompt, so the LLM reads image and words as one sequence."""

    def __init__(self, c: Config):
        super().__init__()
        self.image_token = c.image_token
        self.proj = nn.Linear(c.vit_dim, c.llm_dim)

    def forward(self, patches, ids, embedded):  # patches (images, 256, 1152); ids, embedded: the prompt
        image = self.proj(patches).flatten(0, 1)  # (images · 256, 2048)
        embedded = embedded.clone()
        embedded[ids == self.image_token] = image.to(embedded.dtype)
        return embedded


# %% [llm] Qwen3 layers 1–12 (the middle-layer features)
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rope(x, positions, theta):
    """Rotate pairs (i, i + d/2) of every head by position · theta^(−2i/d). x: (batch, heads, seq, d)."""
    d = x.shape[-1]
    angles = positions[:, None].float() * theta ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    cos, sin = torch.cat([angles, angles], -1).cos(), torch.cat([angles, angles], -1).sin()
    swapped = torch.cat([-x[..., d // 2 :], x[..., : d // 2]], dim=-1)
    return x * cos + swapped * sin


class LLMLayer(nn.Module):
    """Qwen3 decoder layer: RMSNorm → causal grouped-query attention with RMSNorm on each query and key head
    and RoPE → add; RMSNorm → SwiGLU MLP → add."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.norm1 = RMSNorm(c.llm_dim)
        self.q = nn.Linear(c.llm_dim, c.llm_heads * c.llm_head_dim, bias=False)
        self.k = nn.Linear(c.llm_dim, c.llm_kv_heads * c.llm_head_dim, bias=False)
        self.v = nn.Linear(c.llm_dim, c.llm_kv_heads * c.llm_head_dim, bias=False)
        self.q_norm, self.k_norm = RMSNorm(c.llm_head_dim), RMSNorm(c.llm_head_dim)
        self.out = nn.Linear(c.llm_heads * c.llm_head_dim, c.llm_dim, bias=False)
        self.norm2 = RMSNorm(c.llm_dim)
        self.gate = nn.Linear(c.llm_dim, c.llm_mlp, bias=False)
        self.up = nn.Linear(c.llm_dim, c.llm_mlp, bias=False)
        self.down = nn.Linear(c.llm_mlp, c.llm_dim, bias=False)

    def forward(self, x):  # (batch, seq, 2048)
        c, n = self.c, x.shape[1]
        positions = torch.arange(n)
        h = self.norm1(x)
        q = rope(self.q_norm(self.q(h).unflatten(-1, (c.llm_heads, -1))).transpose(1, 2), positions, c.rope_theta)
        k = rope(self.k_norm(self.k(h).unflatten(-1, (c.llm_kv_heads, -1))).transpose(1, 2), positions, c.rope_theta)
        v = self.v(h).unflatten(-1, (c.llm_kv_heads, -1)).transpose(1, 2)
        group = c.llm_heads // c.llm_kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(c.llm_head_dim)
        future = torch.ones(n, n, dtype=torch.bool).triu(1)  # a token reads only itself and earlier tokens
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        x = x + self.out((weights @ v).transpose(1, 2).flatten(2))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class LanguageModel(nn.Module):
    """The first 12 layers of Qwen3-1.7B. GR00T takes hidden_states[12]: the LLM's final RMSNorm applied to the
    output of layer 12 (the other 16 layers are not even in the checkpoint)."""

    def __init__(self, c: Config):
        super().__init__()
        self.embed = nn.Embedding(c.vocab, c.llm_dim)
        self.layers = nn.ModuleList(LLMLayer(c) for _ in range(c.llm_layers))
        self.norm = RMSNorm(c.llm_dim)

    def forward(self, x):  # (batch, seq, 2048), the embedded prompt with the image spliced in
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# %% [vl-adapter] VL adapter: LayerNorm + 4 self-attention blocks
class Attention(nn.Module):
    """Multi-head attention with biases (diffusers' Attention): queries from x, keys and values from \`context\`
    (x itself unless given), 1/√(head dim) scaling, no mask."""

    def __init__(self, d, heads, context_dim=None):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(context_dim or d, d)
        self.v = nn.Linear(context_dim or d, d)
        self.out = nn.Linear(d, d)

    def forward(self, x, context=None):
        context = x if context is None else context
        q = self.q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)  # (batch, heads, n, d/heads)
        k = self.k(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)  # (batch, heads, m, d/heads)
        v = self.v(context).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)  # (batch, heads, n, m)
        return self.out((weights @ v).transpose(1, 2).flatten(2))


class FeedForward(nn.Module):
    """Linear to 4× the width, GELU(tanh), Linear back."""

    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class AdapterBlock(nn.Module):
    """LayerNorm → self-attention → add; LayerNorm → feed-forward → add."""

    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attention = Attention(d, heads)
        self.norm3 = nn.LayerNorm(d)
        self.mlp = FeedForward(d)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm3(x))


class VLAdapter(nn.Module):
    """The action head's own pass over the frozen VLM tokens before the DiT reads them: a LayerNorm (vlln) and
    4 self-attention blocks, 2048 wide. These weights are trained with the action head."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.LayerNorm(c.llm_dim)
        self.blocks = nn.ModuleList(AdapterBlock(c.llm_dim, c.adapter_heads) for _ in range(c.adapter_layers))

    def forward(self, vl):  # (batch, seq, 2048)
        x = self.norm(vl)
        for block in self.blocks:
            x = block(x)
        return x


# %% [state-encoder] State encoder, one per embodiment
class EmbodimentLinear(nn.Module):
    """A Linear per embodiment: a bank of 32 weight matrices; each sample uses the one its robot's id picks."""

    def __init__(self, slots, d_in, d_out):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(slots, d_in, d_out))
        self.b = nn.Parameter(torch.zeros(slots, d_out))

    def forward(self, x, embodiment):  # x: (batch, n, d_in); embodiment: (batch,) slot ids
        return torch.bmm(x, self.W[embodiment]) + self.b[embodiment][:, None]


class EmbodimentMLP(nn.Module):
    def __init__(self, slots, d_in, d_hidden, d_out):
        super().__init__()
        self.layer1 = EmbodimentLinear(slots, d_in, d_hidden)
        self.layer2 = EmbodimentLinear(slots, d_hidden, d_out)

    def forward(self, x, embodiment):
        return self.layer2(F.relu(self.layer1(x, embodiment)), embodiment)


def state_encoder(c: Config):
    """The robot's joint state (sin and cos of each angle, zero-padded to 64) → one 1536-wide token."""
    return EmbodimentMLP(c.embodiments, c.max_state, c.encoder_hidden, c.d_model)


# %% [action-encoder] Action encoder, one per embodiment
def sinusoid(t, d):
    """d/2 sines then d/2 cosines of t at frequencies 10000^(−i/(d/2)). t: any shape → (..., d)."""
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(d // 2, dtype=torch.float32) / (d // 2))
    angles = t.float()[..., None] * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ActionEncoder(nn.Module):
    """Each noisy action (32 numbers) → a 1536-wide token that also knows the flow time:
    W₃ swish(W₂ [W₁ a, sinusoid(1000 t)]), all three Linears picked by the embodiment."""

    def __init__(self, c: Config):
        super().__init__()
        self.d = c.d_model
        self.W1 = EmbodimentLinear(c.embodiments, c.max_action, c.d_model)
        self.W2 = EmbodimentLinear(c.embodiments, 2 * c.d_model, c.d_model)
        self.W3 = EmbodimentLinear(c.embodiments, c.d_model, c.d_model)

    def forward(self, actions, timestep, embodiment):  # actions (batch, 16, 32); timestep (batch,) in 0…999
        tau = sinusoid(timestep[:, None].expand(-1, actions.shape[1]), self.d)  # the same for all 16 rows
        x = torch.cat([self.W1(actions, embodiment), tau], dim=-1)
        x = self.W2(x, embodiment)
        return self.W3(x * torch.sigmoid(x), embodiment)


# %% [timestep] Timestep embedding
class TimestepEmbedding(nn.Module):
    """int(1000 t) → cos and sin at 128 frequencies → Linear, SiLU, Linear → temb, 1536 wide. Every DiT block
    reads temb through its adaptive LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        self.frequency_dim = c.frequency_dim
        self.linear_1 = nn.Linear(c.frequency_dim, c.d_model)
        self.linear_2 = nn.Linear(c.d_model, c.d_model)

    def forward(self, timestep):  # (batch,)
        half = self.frequency_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / (half - 1))
        angles = timestep.float()[:, None] * frequencies
        e = torch.cat([angles.cos(), angles.sin()], dim=-1)
        return self.linear_2(F.silu(self.linear_1(e)))


# %% [adaln] Adaptive LayerNorm
class AdaLayerNorm(nn.Module):
    """LayerNorm with no learned scale or shift; temb supplies them: x̂ (1 + γ) + β with [γ, β] = W SiLU(temb) + b.
    Each block has its own W, so the same t moves each block's tokens differently."""

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.linear = nn.Linear(d, 2 * d)
        self.eps = eps

    def forward(self, x, temb):  # x (batch, n, d); temb (batch, d)
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=1)
        return F.layer_norm(x, x.shape[-1:], eps=self.eps) * (1 + scale[:, None]) + shift[:, None]


# %% [attention] Cross- and self-attention in the DiT
def dit_attention(c: Config, cross: bool):
    """32 heads of 48. A cross-attention block reads keys and values from the 2048-wide VLM tokens; a
    self-attention block reads the 49 state, future and action tokens themselves."""
    return Attention(c.d_model, c.dit_heads, c.llm_dim if cross else None)


# %% [block] DiT block
class DiTBlock(nn.Module):
    """adaLN(x, t) → attention → add; LayerNorm → feed-forward → add. Even blocks cross-attend to the VLM
    tokens, odd blocks self-attend; the timestep enters only through the first norm."""

    def __init__(self, c: Config, cross: bool):
        super().__init__()
        self.cross = cross
        self.norm1 = AdaLayerNorm(c.d_model)
        self.attention = dit_attention(c, cross)
        self.mlp = FeedForward(c.d_model)

    def forward(self, x, vl, temb):
        x = x + self.attention(self.norm1(x, temb), vl if self.cross else None)
        return x + self.mlp(F.layer_norm(x, x.shape[-1:], eps=1e-5))


# %% [dit] The diffusion transformer
class DiT(nn.Module):
    """16 blocks over the 49 tokens, then a last adaptive LayerNorm (shift and scale from temb) and a Linear
    to 1024."""

    def __init__(self, c: Config):
        super().__init__()
        self.timestep = TimestepEmbedding(c)
        self.blocks = nn.ModuleList(DiTBlock(c, cross=i % 2 == 0) for i in range(c.dit_layers))
        self.proj_out_1 = nn.Linear(c.d_model, 2 * c.d_model)
        self.proj_out_2 = nn.Linear(c.d_model, c.encoder_hidden)

    def forward(self, x, vl, timestep):  # x (batch, 49, 1536); vl (batch, seq, 2048); timestep (batch,)
        temb = self.timestep(timestep)
        for block in self.blocks:
            x = block(x, vl, temb)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)  # shift first here, scale first in adaLN
        x = F.layer_norm(x, x.shape[-1:], eps=1e-6) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(x)


# %% [action-decoder] Action decoder, one per embodiment
def action_decoder(c: Config):
    """Each DiT output token (1024) → 32 numbers; the last 16 tokens' outputs are the velocity of the chunk."""
    return EmbodimentMLP(c.embodiments, c.encoder_hidden, c.encoder_hidden, c.max_action)


# %% [normalization] State and action normalization
def state_features(joints):
    """Joint angles (radians) → [sin, cos] of each, per body part; GR00T's GR1 config does this for the state."""
    return torch.cat([joints.sin(), joints.cos()], dim=-1)


def normalize_action(a, low, high):
    """Min-max to [-1, 1] per joint, with the dataset's per-joint min and max (0 where min = max)."""
    span = high - low
    return torch.where(span != 0, 2 * (a - low) / torch.where(span != 0, span, 1) - 1, torch.zeros_like(a))


def unnormalize_action(x, low, high):
    """The model's output back to joint targets."""
    return (x + 1) / 2 * (high - low) + low


# %% [flow-matching] Flow matching: four Euler steps
class ActionHead(nn.Module):
    """System 1. Noise x₀ ~ N(0, I), 16 × 32; at t = 0, ¼, ½, ¾ the DiT predicts a velocity v and the chunk
    moves x ← x + ¼ v. Training interpolated x_t = (1 − t) noise + t action and regressed v = action − noise."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.adapter = VLAdapter(c)
        self.state_encoder = state_encoder(c)
        self.action_encoder = ActionEncoder(c)
        self.position = nn.Embedding(c.max_positions, c.d_model)
        self.future = nn.Embedding(c.future_tokens, c.d_model)
        self.dit = DiT(c)
        self.action_decoder = action_decoder(c)

    def velocity(self, x, t, vl, state, embodiment):
        """One DiT call: (batch, 16, 32) actions at flow time t → (batch, 16, 32) velocity."""
        c, batch = self.c, x.shape[0]
        timestep = torch.full((batch,), int(t * c.timestep_buckets))
        actions = self.action_encoder(x, timestep, embodiment) + self.position.weight[: c.horizon]
        future = self.future.weight.expand(batch, -1, -1)
        tokens = torch.cat([state, future, actions], dim=1)  # (batch, 1 + 32 + 16, 1536)
        out = self.dit(tokens, vl, timestep)
        return self.action_decoder(out, embodiment)[:, -c.horizon :]

    def forward(self, vl, state, embodiment, noise):  # vl from System 2; state (batch, 1, 64); noise (batch, 16, 32)
        vl = self.adapter(vl)
        state = self.state_encoder(state, embodiment)  # computed once, reused by every step
        x, dt = noise, 1.0 / self.c.steps
        for k in range(self.c.steps):
            x = x + dt * self.velocity(x, k * dt, vl, state, embodiment)
        return x


# %% [model] GR00T N1.5
class Eagle(nn.Module):
    """System 2: the camera frame and the instruction → one 2048-wide vector per prompt token."""

    def __init__(self, c: Config):
        super().__init__()
        self.vision = VisionTower(c)
        self.connector = Connector(c)
        self.llm = LanguageModel(c)

    def forward(self, pixels, ids):  # pixels (images, 3, 224, 224); ids (batch, seq) with 256 <IMG_CONTEXT> per image
        embedded = self.connector(self.vision(pixels), ids, self.llm.embed(ids))
        return self.llm(embedded)


class GR00T(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.eagle = Eagle(c)
        self.action_head = ActionHead(c)

    @torch.no_grad()
    def forward(self, pixels, ids, state, embodiment, noise):
        """→ a chunk of 16 normalized actions, 32 wide (the robot's own joints first, zeros after)."""
        return self.action_head(self.eagle(pixels, ids), state, embodiment, noise)
`;export{e as default};
