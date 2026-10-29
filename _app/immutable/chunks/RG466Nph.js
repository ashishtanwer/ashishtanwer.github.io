const n=`# ACT, the Action Chunking Transformer (Zhao et al. 2023), as LeRobot runs lerobot/act_aloha_sim_transfer_cube_human,
# written as plain PyTorch blocks. A top-camera image and 14 joint positions in; a chunk of 100 joint targets out.
# Each "# %% [id]" section is one part of the ACT tab.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    joints: int = 14  # 6 joints + 1 gripper per arm, two arms
    dim: int = 512
    heads: int = 8
    ffn: int = 3200
    encoder_layers: int = 4
    decoder_layers: int = 1  # the original code built 7 but used only the first
    vae_layers: int = 4
    latent: int = 32  # the style variable z
    chunk: int = 100  # actions predicted per query
    n_action_steps: int = 100  # actions played before the next query
    ensemble_coeff: float | None = None  # temporal ensembling: off in this checkpoint (the paper used 0.01)
    image_mean: tuple = (0.485, 0.456, 0.406)  # ImageNet's, stored in the checkpoint
    image_std: tuple = (0.229, 0.224, 0.225)


# %% [inputs] Observation and normalisation
def normalise_inputs(image, joints, joint_mean, joint_std, c: Config, eps=1e-8):
    """image: (batch, 3, 480, 640) uint8 from the top camera; joints: (batch, 14) radians and gripper openings.
    Both become zero-mean, unit-variance numbers with statistics stored in the checkpoint."""
    x = image.float() / 255
    mean, std = torch.tensor(c.image_mean).view(1, 3, 1, 1), torch.tensor(c.image_std).view(1, 3, 1, 1)
    return (x - mean) / (std + eps), (joints - joint_mean) / (joint_std + eps)


def unnormalise_actions(actions, action_mean, action_std):
    """The model's (batch, chunk, 14) outputs back to joint targets the simulator's controllers follow."""
    return actions * action_std + action_mean


# %% [backbone] ResNet-18 and the image tokens
class FrozenBatchNorm(nn.Module):
    """BatchNorm with fixed statistics: a per-channel scale and shift."""

    def __init__(self, channels: int):
        super().__init__()
        for name, value in (("weight", 1.0), ("bias", 0.0), ("running_mean", 0.0), ("running_var", 1.0)):
            self.register_buffer(name, torch.full((channels,), value))

    def forward(self, x):
        scale = self.weight * (self.running_var + 1e-5).rsqrt()
        return x * scale[:, None, None] + (self.bias - self.running_mean * scale)[:, None, None]


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = FrozenBatchNorm(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = FrozenBatchNorm(c_out)
        self.downsample = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), FrozenBatchNorm(c_out)) if stride != 1 else None
        )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + (x if self.downsample is None else self.downsample(x)))


class ResNet18(nn.Module):
    """Stride 2 stem, a max-pool and four layers of two blocks: 480 × 640 → 15 × 20 cells of 512 channels."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = FrozenBatchNorm(64)
        widths = (64, 128, 256, 512)
        for i, width in enumerate(widths):
            c_in = widths[max(i - 1, 0)]
            stride = 1 if i == 0 else 2
            setattr(self, f"layer{i + 1}", nn.Sequential(BasicBlock(c_in, width, stride), BasicBlock(width, width, 1)))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 3, 2, 1)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def position_2d(height: int, width: int, dim: int):
    """Fixed sine-cosine positions for a height × width grid: half the channels encode the row, half the column,
    each index scaled to (0, 2π]."""
    half = dim // 2
    rows = torch.arange(1, height + 1, dtype=torch.float32)[:, None].expand(height, width)
    cols = torch.arange(1, width + 1, dtype=torch.float32)[None, :].expand(height, width)
    rows, cols = rows / (height + 1e-6) * 2 * math.pi, cols / (width + 1e-6) * 2 * math.pi
    freq = 10000 ** (2 * (torch.arange(half) // 2) / half)

    def encode(p):
        p = p[..., None] / freq  # (height, width, half)
        return torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(2)

    return torch.cat((encode(rows), encode(cols)), dim=-1).permute(2, 0, 1)  # (dim, height, width)


class ImageTokens(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.backbone = ResNet18()
        self.project = nn.Conv2d(512, c.dim, 1)

    def forward(self, image):  # (batch, 3, H, W), normalised
        features = self.backbone(image)  # (batch, 512, H/32, W/32)
        pos = position_2d(features.shape[2], features.shape[3], self.project.out_channels)
        tokens = self.project(features).flatten(2).transpose(1, 2)  # (batch, cells, dim): one token per cell
        return tokens, pos.flatten(1).T  # positions: (cells, dim)


# %% [encoder] Transformer encoder layer
class Attention(nn.Module):
    """Multi-head attention with one packed input projection, as nn.MultiheadAttention stores it."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value):  # (batch, length, dim) each
        b, n, d = query.shape
        wq, wk, wv = self.in_proj_weight.chunk(3)
        bq, bk, bv = self.in_proj_bias.chunk(3)
        split = lambda x: x.view(b, -1, self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        q, k, v = split(F.linear(query, wq, bq)), split(F.linear(key, wk, bk)), split(F.linear(value, wv, bv))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)  # (b, heads, n, keys)
        return self.out_proj((weights @ v).transpose(1, 2).reshape(b, n, d))


class EncoderLayer(nn.Module):
    """Post-norm: attention, add, LayerNorm; ReLU MLP, add, LayerNorm. Positions join the queries and keys
    only, so every layer sees them afresh and the values stay position-free."""

    def __init__(self, c: Config):
        super().__init__()
        self.self_attn = Attention(c.dim, c.heads)
        self.linear1, self.linear2 = nn.Linear(c.dim, c.ffn), nn.Linear(c.ffn, c.dim)
        self.norm1, self.norm2 = nn.LayerNorm(c.dim), nn.LayerNorm(c.dim)

    def forward(self, x, pos):
        q = k = x + pos
        x = self.norm1(x + self.self_attn(q, k, x))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


# %% [latent] The style variable z
class VaeEncoder(nn.Module):
    """Training only: [CLS, joints, the next 100 demonstrated actions] → 4 encoder layers → μ and log σ² of z."""

    def __init__(self, c: Config):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, c.dim))
        self.joints_in = nn.Linear(c.joints, c.dim)
        self.actions_in = nn.Linear(c.joints, c.dim)
        self.layers = nn.ModuleList(EncoderLayer(c) for _ in range(c.vae_layers))
        self.to_latent = nn.Linear(c.dim, 2 * c.latent)
        self.latent = c.latent
        self.register_buffer("pos", sinusoid_1d(c.chunk + 2, c.dim), persistent=False)  # fixed, not learned

    def forward(self, joints, actions, actions_is_pad):
        b = joints.shape[0]
        x = torch.cat([self.cls.expand(b, 1, -1), self.joints_in(joints)[:, None], self.actions_in(actions)], 1)
        pad = torch.cat([torch.zeros(b, 2, dtype=torch.bool), actions_is_pad], 1)
        for layer in self.layers:
            x = masked_layer(layer, x, self.pos, pad)
        params = self.to_latent(x[:, 0])
        return params[:, : self.latent], params[:, self.latent :]  # μ, log σ²


def sinusoid_1d(length: int, dim: int):
    position = torch.arange(length, dtype=torch.float64)[:, None]
    angle = position / torch.pow(10000, 2 * (torch.arange(dim) // 2) / dim)
    table = torch.zeros(length, dim, dtype=torch.float64)
    table[:, 0::2], table[:, 1::2] = angle[:, 0::2].sin(), angle[:, 1::2].cos()
    return table.float()


def masked_layer(layer, x, pos, pad):
    """An encoder layer whose attention skips padded actions (past the end of the demonstration)."""
    att, d, h = layer.self_attn, x.shape[-1], layer.self_attn.heads
    q = k = x + pos
    wq, wk, wv = att.in_proj_weight.chunk(3)
    bq, bk, bv = att.in_proj_bias.chunk(3)
    split = lambda t: t.view(x.shape[0], -1, h, d // h).transpose(1, 2)  # noqa: E731
    scores = split(F.linear(q, wq, bq)) @ split(F.linear(k, wk, bk)).transpose(-1, -2) / math.sqrt(d // h)
    weights = torch.softmax(scores.masked_fill(pad[:, None, None], float("-inf")), dim=-1)
    out = att.out_proj((weights @ split(F.linear(x, wv, bv))).transpose(1, 2).reshape(x.shape))
    x = layer.norm1(x + out)
    return layer.norm2(x + layer.linear2(F.relu(layer.linear1(x))))


def z_token(project_z: nn.Linear, mu=None, log_var=None, batch=1):
    """At run time there are no future actions to encode, so z = 0 and the token is the projection's bias.
    In training z = μ + σ·ε is sampled from the encoder's distribution."""
    if mu is None:
        z = torch.zeros(batch, project_z.in_features)
    else:
        z = mu + (log_var / 2).exp() * torch.randn_like(mu)
    return project_z(z)


# %% [decoder] Transformer decoder layer
class DecoderLayer(nn.Module):
    """100 queries, one per future step: self-attention among them, cross-attention into the encoder's
    memory, a ReLU MLP; each followed by add and LayerNorm."""

    def __init__(self, c: Config):
        super().__init__()
        self.self_attn = Attention(c.dim, c.heads)
        self.multihead_attn = Attention(c.dim, c.heads)
        self.linear1, self.linear2 = nn.Linear(c.dim, c.ffn), nn.Linear(c.ffn, c.dim)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(c.dim), nn.LayerNorm(c.dim), nn.LayerNorm(c.dim)

    def forward(self, x, query_pos, memory, memory_pos):
        q = k = x + query_pos
        x = self.norm1(x + self.self_attn(q, k, x))
        x = self.norm2(x + self.multihead_attn(x + query_pos, memory + memory_pos, memory))
        return self.norm3(x + self.linear2(F.relu(self.linear1(x))))


# %% [action-head] Action head
class ActionHead(nn.Module):
    """One Linear per decoder output: 512 numbers → 14 normalised joint targets."""

    def __init__(self, c: Config):
        super().__init__()
        self.linear = nn.Linear(c.dim, c.joints)

    def forward(self, decoded):  # (batch, chunk, dim) → (batch, chunk, 14)
        return self.linear(decoded)


# %% [chunking] Playing a chunk, or ensembling overlapping chunks
def play_chunks(query, observations, c: Config):
    """Query the model once, then play n_action_steps of its chunk open loop before querying again.
    \`query(observation)\` returns a (chunk, joints) tensor; one action comes out per step."""
    queue = []
    for observation in observations:
        if not queue:
            queue = list(query(observation)[: c.n_action_steps])
        yield queue.pop(0)


def temporal_ensemble(chunks, coeff):
    """ACT's temporal ensembling (Algorithm 2): the model is queried every step; the action for step t is the
    weighted mean of every chunk that reaches t, wᵢ = exp(−coeff · i) with i = 0 for the oldest chunk.
    chunks: (steps, chunk, joints), chunks[k] queried at step k."""
    steps, length, _ = chunks.shape
    out = []
    for t in range(steps):
        queried = range(max(0, t - length + 1), t + 1)  # oldest first
        values = torch.stack([chunks[k, t - k] for k in queried])
        weights = torch.exp(-coeff * torch.arange(len(values), dtype=values.dtype))
        out.append((weights[:, None] * values).sum(0) / weights.sum())
    return torch.stack(out)


# %% [model] ACT at run time
class ACT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.image = ImageTokens(c)
        self.project_z = nn.Linear(c.latent, c.dim)
        self.joints_in = nn.Linear(c.joints, c.dim)
        self.token_pos = nn.Parameter(torch.zeros(2, c.dim))  # learned positions of the z and joint tokens
        self.encoder = nn.ModuleList(EncoderLayer(c) for _ in range(c.encoder_layers))
        self.query_pos = nn.Parameter(torch.zeros(c.chunk, c.dim))  # the 100 learned decoder queries
        self.decoder = nn.ModuleList(DecoderLayer(c) for _ in range(c.decoder_layers))
        self.decoder_norm = nn.LayerNorm(c.dim)
        self.head = ActionHead(c)

    def forward(self, image, joints):  # normalised (batch, 3, H, W) and (batch, 14)
        b = image.shape[0]
        cells, cell_pos = self.image(image)
        tokens = torch.cat([z_token(self.project_z, batch=b)[:, None], self.joints_in(joints)[:, None], cells], 1)
        pos = torch.cat([self.token_pos, cell_pos])  # (2 + cells, dim)
        memory = tokens
        for layer in self.encoder:
            memory = layer(memory, pos)
        x = torch.zeros(b, self.c.chunk, self.c.dim)  # the queries start empty; only their positions differ
        for layer in self.decoder:
            x = layer(x, self.query_pos, memory, pos)
        return self.head(self.decoder_norm(x))  # (batch, 100, 14) normalised joint targets
`;export{n as default};
