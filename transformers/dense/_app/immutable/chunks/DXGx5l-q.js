const e=`# V-JEPA 2, the video world model in the V-JEPA 2 tab, and V-JEPA 2-AC, its action-conditioned predictor, written as
# plain PyTorch blocks. A ViT cuts a clip into 2 × 16 × 16 tubelets and encodes them with 3-D RoPE. Pretraining hides
# blocks of tubelets, encodes the rest, and trains a predictor to output the latents a slowly updated copy of the
# encoder gives for the hidden ones: an L1 distance in latent space, so no pixel is ever reconstructed. V-JEPA 2-AC
# freezes the encoder and trains a second predictor that rolls a frame's latents forward under robot actions; a
# planner then searches actions with the cross-entropy method. Each "# %% [id]" section is one part of the tab.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    crop: int = 256  # frames are 256 × 256
    patch: int = 16  # 16 × 16 pixels per tubelet
    tubelet: int = 2  # 2 frames per tubelet
    d_model: int = 1024  # ViT-L
    n_heads: int = 16  # 64 dimensions per head
    n_layers: int = 24
    d_mlp: int = 4096
    d_pred: int = 384  # the predictor is a narrower ViT
    pred_heads: int = 12
    pred_layers: int = 12
    mask_tokens: int = 10  # learned starting points for hidden tubelets; pretraining only ever uses token 0
    eps: float = 1e-6


@dataclass
class ACConfig:
    d_model: int = 1408  # V-JEPA 2's ViT-g encoder, frozen: 40 layers, 22 heads
    d_pred: int = 1024
    pred_heads: int = 16
    pred_layers: int = 24
    action_dim: int = 7  # Δx, Δy, Δz, Δ rotation (euler xyz), Δ gripper; the state is the same 7 numbers absolute
    grid: int = 16  # 16 × 16 tokens per frame
    eps: float = 1e-6


# %% [tubelets] Tubelet embedding
class TubeletEmbedding(nn.Module):
    """A 3-D convolution with kernel = stride = 2 × 16 × 16: each tubelet (2 frames of a 16 × 16 patch, 1,536 numbers)
    becomes one d_model vector. A 16-frame 256 × 256 clip gives 8 × 16 × 16 = 2,048 tokens, frame by frame, row by row.
    A still image is repeated into 2 frames first."""

    def __init__(self, c: Config):
        super().__init__()
        size = (c.tubelet, c.patch, c.patch)
        self.proj = nn.Conv3d(3, c.d_model, kernel_size=size, stride=size)

    def forward(self, video):  # video: (batch, 3, frames, height, width), normalised with ImageNet's mean and std
        return self.proj(video).flatten(2).transpose(1, 2)  # (batch, tokens, d_model)


# %% [rope] 3-D rotary position embedding
def token_positions(ids, grid):
    """Time slice, row and column of each token id (ids count frame by frame, row by row)."""
    per_frame = grid * grid
    return ids // per_frame, (ids % per_frame) // grid, ids % grid


def rotate(x, position):
    """Turn x (..., tokens, D) by angle position × ω_i, ω_i = 10000^(−i / (D/2)). As in the released code, the D/2
    angles are laid out twice end to end ([θ_0 … θ_{D/2−1}, θ_0 … θ_{D/2−1}]) while coordinates are paired
    neighbour by neighbour, so the two halves of a pair turn by different angles. The trained weights expect this."""
    d = x.shape[-1]
    omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=x.dtype) / (d / 2))
    angle = position[..., None].to(x.dtype) * omega  # (..., tokens, D/2)
    cos, sin = torch.cat([angle.cos(), angle.cos()], -1), torch.cat([angle.sin(), angle.sin()], -1)
    partner = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)  # (x0, x1) → (−x1, x0)
    return x * cos + partner * sin


def rope3d(x, ids, grid):
    """x: (batch, heads, tokens, 64). The first 20 dimensions turn with the time slice, the next 20 with the row, the
    next 20 with the column; the last 4 are left as they are. Positions follow the ids, so a subset of tokens (the
    context encoder's visible tubelets) keeps its places in the clip."""
    part = 2 * ((x.shape[-1] // 3) // 2)  # 20
    frame, row, col = token_positions(ids, grid)
    turned = [rotate(x[..., i * part : (i + 1) * part], p) for i, p in enumerate((frame, row, col))]
    return torch.cat([*turned, x[..., 3 * part :]], dim=-1)


# %% [attention] Self-attention with 3-D RoPE
class Attention(nn.Module):
    """Every tubelet attends to every other one, in every frame. q and k are rotated by their 3-D positions, so q·k
    depends on how far apart two tubelets are in time, height and width."""

    def __init__(self, d_model, n_heads, grid):
        super().__init__()
        self.n_heads, self.grid = n_heads, grid
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, ids):  # x: (batch, tokens, d_model); ids: (tokens,) positions in the clip
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)  # each (b, heads, n, 64)
        q, k = rope3d(q, ids, self.grid), rope3d(k, ids, self.grid)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # (b, heads, n, n)
        return self.proj((weights @ v).transpose(1, 2).reshape(b, n, d))


# %% [mlp] Feed-forward
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-norm: x + attention(LN(x)), then x + MLP(LN(x)). The encoders and the predictor use the same block."""

    def __init__(self, d_model, n_heads, d_mlp, grid, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.attention = Attention(d_model, n_heads, grid)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x, ids):
        x = x + self.attention(self.norm1(x), ids)
        return x + self.mlp(self.norm2(x))


# %% [encoder] ViT encoder
class Encoder(nn.Module):
    """Tubelets → 24 blocks → LayerNorm. Given \`keep\`, only those tubelets go in (the context encoder in pretraining):
    the hidden ones are not replaced by anything, they are simply absent, and RoPE still places the rest correctly."""

    def __init__(self, c: Config):
        super().__init__()
        self.grid = c.crop // c.patch
        self.embed = TubeletEmbedding(c)
        self.blocks = nn.ModuleList(Block(c.d_model, c.n_heads, c.d_mlp, self.grid, c.eps) for _ in range(c.n_layers))
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, video, keep=None):  # keep: (visible,) token ids, sorted
        x = self.embed(video)
        ids = torch.arange(x.shape[1])
        if keep is not None:
            x, ids = x[:, keep], keep
        for block in self.blocks:
            x = block(x, ids)
        return self.norm(x)  # (batch, tokens or visible, d_model)


# %% [masks] Multi-block masks
def multiblock_mask(grid, blocks, scale, aspect, step=0):
    """Hide the union of \`blocks\` rectangles, each covering \`scale\` of a frame's tubelets with an aspect ratio drawn from
    \`aspect\`, through every time slice (a tube), so the predictor cannot copy a hidden patch from a neighbouring frame.
    The block size comes from a generator seeded with the training step; the positions from torch's global generator.
    Returns (visible ids, hidden ids)."""
    frames, rows, cols = grid
    g = torch.Generator().manual_seed(step)
    torch.rand(1, generator=g), torch.rand(1, generator=g)  # time extent and area: drawn, but fixed (all frames, scale)
    keep = int(rows * cols * scale)
    ratio = aspect[0] + torch.rand(1, generator=g).item() * (aspect[1] - aspect[0])
    h = min(rows, int(round(math.sqrt(keep * ratio))))
    w = min(cols, int(round(math.sqrt(keep / ratio))))
    visible = torch.ones(frames, rows, cols, dtype=torch.int32)
    for _ in range(blocks):
        top = torch.randint(0, rows - h + 1, (1,))
        left = torch.randint(0, cols - w + 1, (1,))
        torch.randint(0, 1, (1,))  # the start frame: the block spans all of them
        visible[:, top : top + h, left : left + w] = 0
    visible = visible.flatten()
    return torch.nonzero(visible).squeeze(1), torch.nonzero(visible == 0).squeeze(1)


# %% [predictor] Predictor
class Predictor(nn.Module):
    """A narrower ViT. The visible tubelets' latents are projected to d_pred; every hidden tubelet starts as the same
    learned mask token. All tokens are put back in clip order, run through 12 blocks with 3-D
    RoPE on their positions, and the hidden ones are projected back to d_model: the prediction."""

    def __init__(self, c: Config):
        super().__init__()
        grid = c.crop // c.patch
        self.embed = nn.Linear(c.d_model, c.d_pred)
        self.mask_tokens = nn.Parameter(torch.zeros(c.mask_tokens, c.d_pred))
        self.blocks = nn.ModuleList(
            Block(c.d_pred, c.pred_heads, 4 * c.d_pred, grid, c.eps) for _ in range(c.pred_layers)
        )
        self.norm = nn.LayerNorm(c.d_pred, eps=c.eps)
        self.proj = nn.Linear(c.d_pred, c.d_model)

    def forward(self, context, visible, hidden, mask_index):  # context: (batch, visible, d_model)
        b = context.shape[0]
        x = torch.cat([self.embed(context), self.mask_tokens[mask_index].expand(b, len(hidden), -1)], dim=1)
        ids = torch.cat([visible, hidden])
        order = ids.argsort()  # clip order
        x = x[:, order]
        for block in self.blocks:
            x = block(x, ids[order])
        x = self.norm(x)[:, order.argsort()]  # back to [visible; hidden]
        return self.proj(x[:, len(visible) :])  # (batch, hidden, d_model)


# %% [jepa] The JEPA objective
def jepa_loss(context_encoder, target_encoder, predictor, video, masks):
    """For each mask: the context encoder sees only the visible tubelets, the predictor fills in the hidden ones, and
    the target encoder (an exponential moving average of the context encoder, never trained directly) sees the whole
    clip. The loss is the mean |prediction − target| over the hidden tubelets' latents, after a plain layer norm of
    the target. Nothing is decoded: the error is measured in latent space, not in pixels. Both masks start from mask
    token 0: the training code picks the token by clip length, and every clip has the same length."""
    with torch.no_grad():
        target = target_encoder(video)
        target = F.layer_norm(target, target.shape[-1:])
    losses = []
    for visible, hidden in masks:
        prediction = predictor(context_encoder(video, keep=visible), visible, hidden, mask_index=0)
        losses.append((prediction - target[:, hidden]).abs().mean())
    return torch.stack(losses).mean()


# %% [ac-predictor] Action-conditioned predictor
class ACAttention(nn.Module):
    """Attention over frames laid out as [action, state, 256 patches] each. Patch tokens are rotated by time slice,
    row and column; the action and state tokens only by time, in their first 20 dimensions."""

    def __init__(self, d_model, n_heads, grid):
        super().__init__()
        self.n_heads, self.grid = n_heads, grid
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, frames, mask):  # x: (batch, frames · (2 + grid²), d_model); mask: (n, n) True = may attend
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        per = 2 + self.grid * self.grid
        slot = torch.arange(n) % per  # 0 action, 1 state, 2 … patches
        frame = torch.arange(n) // per
        patch_ids = frame * self.grid * self.grid + (slot - 2).clamp(min=0)
        part = 2 * ((q.shape[-1] // 3) // 2)

        def place(t):
            patches = rope3d(t, patch_ids, self.grid)
            timed = torch.cat([rotate(t[..., :part], frame), t[..., part:]], dim=-1)
            return torch.where((slot < 2)[:, None], timed, patches)

        q, k = place(q), place(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        return self.proj((weights @ v).transpose(1, 2).reshape(b, n, d))


class ACBlock(nn.Module):
    def __init__(self, d_model, n_heads, grid, eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.attention = ACAttention(d_model, n_heads, grid)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = MLP(d_model, 4 * d_model)

    def forward(self, x, frames, mask):
        x = x + self.attention(self.norm1(x), frames, mask)
        return x + self.mlp(self.norm2(x))


def block_causal(frames, per):
    """Frame t's tokens see every token of frames 0 … t and nothing later."""
    frame = torch.arange(frames * per) // per
    return frame[:, None] >= frame[None, :]


class ACPredictor(nn.Module):
    """Each frame's 256 latents (from the frozen encoder, layer-normed) are projected to d_pred; the robot's action and
    its end-effector state become one token each in front of them. Block-causal attention lets frame t read frames
    0 … t with their actions; the output at frame t's patches is the prediction of frame t + 1's latents."""

    def __init__(self, c: ACConfig):
        super().__init__()
        self.grid = c.grid
        self.embed = nn.Linear(c.d_model, c.d_pred)
        self.action = nn.Linear(c.action_dim, c.d_pred)
        self.state = nn.Linear(c.action_dim, c.d_pred)
        self.blocks = nn.ModuleList(ACBlock(c.d_pred, c.pred_heads, c.grid, c.eps) for _ in range(c.pred_layers))
        self.norm = nn.LayerNorm(c.d_pred, eps=c.eps)
        self.proj = nn.Linear(c.d_pred, c.d_model)

    def forward(self, z, actions, states):  # z: (batch, frames · 256, d_model); actions, states: (batch, frames, 7)
        b, frames = actions.shape[:2]
        x = self.embed(z).view(b, frames, self.grid * self.grid, -1)
        x = torch.cat([self.action(actions)[:, :, None], self.state(states)[:, :, None], x], dim=2).flatten(1, 2)
        mask = block_causal(frames, 2 + self.grid * self.grid)
        for block in self.blocks:
            x = block(x, frames, mask)
        x = x.view(b, frames, -1, x.shape[-1])[:, :, 2:].flatten(1, 2)  # drop the action and state tokens
        return self.proj(self.norm(x))  # (batch, frames · 256, d_model): frame t's row predicts frame t + 1


# %% [rollout] Rolling the latent state forward
def euler_to_matrix(angles):
    """Extrinsic x-y-z euler angles (..., 3) → rotation matrices (..., 3, 3): R = Rz(γ) Ry(β) Rx(α)."""
    a, b, g = angles.unbind(-1)
    ca, sa, cb, sb, cg, sg = a.cos(), a.sin(), b.cos(), b.sin(), g.cos(), g.sin()
    return torch.stack(
        [
            torch.stack([cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa], -1),
            torch.stack([sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa], -1),
            torch.stack([-sb, cb * sa, cb * ca], -1),
        ],
        -2,
    )


def matrix_to_euler(m):
    return torch.stack([torch.atan2(m[..., 2, 1], m[..., 2, 2]), -torch.asin(m[..., 2, 0].clamp(-1, 1)),
                        torch.atan2(m[..., 1, 0], m[..., 0, 0])], -1)  # fmt: skip


def next_pose(pose, action):
    """Where the end effector ends up: positions add, rotations compose (Δ applied after the current one), and the
    gripper's closedness adds and stays in [0, 1]."""
    rotation = euler_to_matrix(action[..., 3:6]) @ euler_to_matrix(pose[..., 3:6])
    gripper = (pose[..., 6:] + action[..., 6:]).clamp(0, 1)
    return torch.cat([pose[..., :3] + action[..., :3], matrix_to_euler(rotation), gripper], dim=-1)


def step(predictor, latents, actions, poses):
    """One step of the world model: latents (batch, t, 256, d) of frames 0 … t − 1 with their actions and poses
    predict frame t; only that frame's output is kept, layer-normed like the encoder's targets."""
    b, t, n, d = latents.shape
    nxt = predictor(latents.flatten(1, 2), actions, poses)[:, -n:]
    return F.layer_norm(nxt, (d,)).view(b, 1, n, d), next_pose(poses[:, -1:], actions[:, -1:])


# %% [cem] Planning with the cross-entropy method
def cem(predictor, latent, pose, goal, rollout=2, samples=400, topk=10, iterations=10, momentum=0.15,
        momentum_std=0.15, maxnorm=0.05):  # fmt: skip
    """Search \`rollout\` actions that bring the predicted latent closest to the goal frame's (an L1 energy):
    1. sample actions from a Gaussian per step (Δx, Δy, Δz within ±maxnorm; Δgripper within ±0.75; no rotation),
    2. roll every sample out with the world model,
    3. keep the topk with the smallest mean |final latent − goal latent|,
    4. move the Gaussian part way to their mean and spread.
    The plan is the final mean; the robot executes its first action and plans again from the next frame."""
    mean = torch.zeros(rollout, 4)
    std = torch.cat([torch.full((rollout, 3), maxnorm), torch.ones(rollout, 1)], dim=1)
    for _ in range(iterations):
        latents, poses, plans = latent.expand(samples, -1, -1, -1), pose.expand(samples, -1, -1), None
        for h in range(rollout):
            a = torch.randn(samples, 4) * std[h] + mean[h]
            a = torch.cat([a[:, :3].clamp(-maxnorm, maxnorm), torch.zeros(samples, 3), a[:, 3:].clamp(-0.75, 0.75)], 1)
            plans = a[:, None] if plans is None else torch.cat([plans, a[:, None]], dim=1)
            nxt, nxt_pose = step(predictor, latents, plans, poses)
            latents, poses = torch.cat([latents, nxt], dim=1), torch.cat([poses, nxt_pose], dim=1)
        energy = (latents[:, -1].flatten(1) - goal.flatten(1)).abs().mean(dim=-1)  # (samples,)
        best = plans[energy.topk(topk, largest=False).indices][..., [0, 1, 2, 6]]  # (topk, rollout, 4)
        mean = best.mean(0) * (1 - momentum) + mean * momentum
        std = best.std(0) * (1 - momentum_std) + std * momentum_std
    gripper = torch.where(mean[:, 3:].abs() < 0.25, torch.zeros_like(mean[:, 3:]), mean[:, 3:])
    return torch.cat([mean[:, :3], torch.zeros(rollout, 3), gripper], dim=1)  # (rollout, 7)


# %% [model] V-JEPA 2 and V-JEPA 2-AC
class VJEPA2(nn.Module):
    """Pretraining: a context encoder, its slowly updated copy as the target encoder, and the predictor."""

    def __init__(self, c: Config):
        super().__init__()
        self.context_encoder = Encoder(c)
        self.target_encoder = Encoder(c)
        self.predictor = Predictor(c)

    def forward(self, video, masks):
        return jepa_loss(self.context_encoder, self.target_encoder, self.predictor, video, masks)


class WorldModel(nn.Module):
    """V-JEPA 2-AC: the frozen encoder reads one frame at a time (repeated into a 2-frame tubelet), the action-conditioned
    predictor imagines what comes next, and cem() plans towards a goal image. Its encoder is V-JEPA 2's ViT-g:
    Config(d_model=1408, n_heads=22, n_layers=40, d_mlp=6144)."""

    def __init__(self, c: Config, ac: ACConfig):
        super().__init__()
        self.encoder = Encoder(c)
        self.predictor = ACPredictor(ac)

    def encode(self, frames):  # frames: (batch, 3, T, height, width) → (batch, T, 256, d), layer-normed
        b, _, t = frames.shape[:3]
        stills = frames.transpose(1, 2).flatten(0, 1)[:, :, None].repeat(1, 1, 2, 1, 1)
        latents = self.encoder(stills).view(b, t, -1, self.predictor.embed.in_features)
        return F.layer_norm(latents, latents.shape[-1:])

    def plan(self, frame, pose, goal_frame, **settings):
        latent, goal = self.encode(frame), self.encode(goal_frame)
        return cem(self.predictor, latent, pose, goal, **settings)
`;export{e as default};
