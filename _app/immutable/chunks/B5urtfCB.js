const e=`# IRIS (Micheli, Alonso & Fleuret 2023), eloialonso/iris pretrained_models/Breakout.pt, as plain PyTorch.
# A discrete autoencoder turns each 64 × 64 frame into 4 × 4 = 16 tokens from a codebook of 512 vectors. A GPT reads
# the frames' tokens and the actions interleaved, 17 tokens per step, and predicts the next frame's tokens one at a
# time, the reward and whether the episode ends. Sampling those predictions plays the game in imagination, and the
# actor-critic learns only there, from λ-returns of imagined rewards.
# Each "# %% [id]" section is one step of the IRIS tab's diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class Config:
    image_size: int = 64
    channels: int = 64  # every level of the autoencoder
    levels: int = 5  # 64, 32, 16, 8 and 4 px
    res_blocks: int = 2  # per level going down; one more going up
    attention_sizes: tuple = (8, 16)  # levels with attention after each ResNet block
    z_channels: int = 512
    vocab: int = 512  # codebook vectors (frame token ids)
    code_dim: int = 512  # width of one codebook vector
    actions: int = 4  # Breakout: NOOP, FIRE, RIGHT, LEFT
    frame_tokens: int = 16  # 4 × 4 tokens per frame
    max_blocks: int = 20  # steps the GPT reads at once
    layers: int = 10
    heads: int = 4
    width: int = 256
    lstm: int = 512
    gamma: float = 0.995
    lambda_: float = 0.95
    entropy_weight: float = 0.001

    @property
    def block(self):  # tokens per step: the frame's 16, then the action
        return self.frame_tokens + 1


def group_norm(channels):
    return nn.GroupNorm(32, channels, eps=1e-6)


# %% [encoder] Encoder: frame → 4 × 4 × 512
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1, self.conv1 = group_norm(ch), nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2, self.conv2 = group_norm(ch), nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(h)))


class PixelAttention(nn.Module):
    """One head over the pixels of a feature map; the residual is added to the un-normalized input."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = group_norm(ch)
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(ch, ch, 1) for _ in range(4))

    def forward(self, x):
        b, ch, h, w = x.shape
        y = self.norm(x)
        q, k, v = (f(y).flatten(2) for f in (self.q, self.k, self.v))  # (b, ch, h·w)
        weights = torch.softmax(q.transpose(1, 2) @ k / ch**0.5, dim=-1)  # (b, h·w, h·w)
        return x + self.proj_out((v @ weights.transpose(1, 2)).view(b, ch, h, w))


class Encoder(nn.Module):
    """Two ResNet blocks per level, attention at 16 and 8 px, a stride-2 convolution between levels (64 → 4 px),
    then a middle block and a 1 × 1 projection to the 512-wide vectors the codebook is searched with."""

    def __init__(self, c: Config):
        super().__init__()
        ch, sizes = c.channels, [c.image_size >> i for i in range(c.levels)]
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        self.blocks = nn.ModuleList(nn.ModuleList(ResBlock(ch) for _ in range(c.res_blocks)) for _ in sizes)
        self.attention = nn.ModuleList(
            nn.ModuleList(PixelAttention(ch) if s in c.attention_sizes else nn.Identity() for _ in range(c.res_blocks))
            for s in sizes
        )
        self.downsamples = nn.ModuleList(nn.Conv2d(ch, ch, 3, stride=2) for _ in sizes[:-1])
        self.mid = nn.Sequential(ResBlock(ch), PixelAttention(ch), ResBlock(ch))
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, c.z_channels, 3, padding=1)
        self.pre_quant = nn.Conv2d(c.z_channels, c.code_dim, 1)

    def forward(self, frames):  # frames: (b, 3, 64, 64) in [0, 1]
        h = self.conv_in(frames * 2 - 1)
        for level, (blocks, attention) in enumerate(zip(self.blocks, self.attention)):
            for block, attn in zip(blocks, attention):
                h = attn(block(h))
            if level < len(self.downsamples):
                h = self.downsamples[level](F.pad(h, (0, 1, 0, 1)))  # pad right and bottom, then stride 2
        h = self.conv_out(F.silu(self.norm_out(self.mid(h))))
        return self.pre_quant(h)  # z: (b, 512, 4, 4)


# %% [quantize] Nearest codebook vector → token ids
def quantize(z, codebook):
    """Each of the 16 vectors of z is replaced by its nearest codebook vector (squared distance); the index of that
    vector is the token. codebook: (512, 512)."""
    b, e, h, w = z.shape
    flat = z.permute(0, 2, 3, 1).reshape(-1, e)  # (b·16, 512), row by row
    distances = flat.pow(2).sum(1, keepdim=True) + codebook.pow(2).sum(1) - 2 * flat @ codebook.t()
    tokens = distances.argmin(-1)  # (b·16,)
    z_q = codebook[tokens].view(b, h, w, e).permute(0, 3, 1, 2)
    return tokens.view(b, h * w), z_q, distances.view(b, h * w, -1)


# %% [decoder] Decoder: tokens → frame
class Decoder(nn.Module):
    """The encoder in reverse: codebook vectors → a 1 × 1 projection → a middle block → three ResNet blocks per level,
    nearest-neighbour ×2 and a 3 × 3 convolution between levels (4 → 64 px) → RGB."""

    def __init__(self, c: Config):
        super().__init__()
        ch, sizes = c.channels, [c.image_size >> i for i in reversed(range(c.levels))]  # 4 … 64
        self.post_quant = nn.Conv2d(c.code_dim, c.z_channels, 1)
        self.conv_in = nn.Conv2d(c.z_channels, ch, 3, padding=1)
        self.mid = nn.Sequential(ResBlock(ch), PixelAttention(ch), ResBlock(ch))
        self.blocks = nn.ModuleList(nn.ModuleList(ResBlock(ch) for _ in range(c.res_blocks + 1)) for _ in sizes)
        self.attention = nn.ModuleList(
            nn.ModuleList(
                PixelAttention(ch) if s in c.attention_sizes else nn.Identity() for _ in range(c.res_blocks + 1)
            )
            for s in sizes
        )
        self.upsamples = nn.ModuleList(nn.Conv2d(ch, ch, 3, padding=1) for _ in sizes[:-1])
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z_q):  # (b, 512, 4, 4)
        h = self.mid(self.conv_in(self.post_quant(z_q)))
        for level, (blocks, attention) in enumerate(zip(self.blocks, self.attention)):
            for block, attn in zip(blocks, attention):
                h = attn(block(h))
            if level < len(self.upsamples):
                h = self.upsamples[level](F.interpolate(h, scale_factor=2.0, mode='nearest'))
        out = self.conv_out(F.silu(self.norm_out(h)))  # (b, 3, 64, 64) in about [-1, 1]
        return ((out + 1) / 2).clamp(0, 1)


class Tokenizer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.encoder, self.decoder = Encoder(c), Decoder(c)
        self.codebook = nn.Parameter(torch.zeros(c.vocab, c.code_dim))

    def encode(self, frames):  # → (b, 16) token ids
        return quantize(self.encoder(frames), self.codebook)[0]

    def decode(self, tokens):  # (b, 16) → (b, 3, 64, 64)
        side = int(tokens.shape[1] ** 0.5)
        return self.decoder(self.codebook[tokens].view(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2))


# %% [embedding] One sequence: frame tokens and actions
class Embedding(nn.Module):
    """Step t is a block of 17 tokens: frame t's 16 token ids, then the action taken at frame t. Frame tokens and
    actions have their own tables; a learned position (0 … 339) is added."""

    def __init__(self, c: Config):
        super().__init__()
        self.block = c.block
        self.frame = nn.Embedding(c.vocab, c.width)
        self.action = nn.Embedding(c.actions, c.width)
        self.position = nn.Embedding(c.max_blocks * c.block, c.width)

    def forward(self, tokens, start: int = 0):  # tokens: (b, t), the ids at positions start … start + t − 1
        positions = start + torch.arange(tokens.shape[1], device=tokens.device)
        is_action = positions % self.block == self.block - 1
        x = self.position(positions).expand(tokens.shape[0], -1, -1).clone()
        x[:, is_action] += self.action(tokens[:, is_action])
        x[:, ~is_action] += self.frame(tokens[:, ~is_action])
        return x  # (b, t, 256)


# %% [attention] Causal self-attention
class SelfAttention(nn.Module):
    """4 heads of 64. Every token sees itself and everything before it. \`cache\` keeps the keys and values of the
    tokens already read, so imagination feeds one new token per call."""

    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.query, self.key, self.value, self.proj = (nn.Linear(c.width, c.width) for _ in range(4))

    def forward(self, x, cache: dict | None = None):
        b, t, width = x.shape
        q, k, v = (f(x).view(b, t, self.heads, -1).transpose(1, 2) for f in (self.query, self.key, self.value))
        if cache is not None:
            if cache:
                k, v = torch.cat([cache['k'], k], dim=2), torch.cat([cache['v'], v], dim=2)
            cache['k'], cache['v'] = k, v
        past = k.shape[2] - t
        visible = torch.ones(t, past + t, dtype=torch.bool, device=x.device).tril(past)
        scores = (q @ k.transpose(-2, -1)) / k.shape[-1] ** 0.5  # (b, 4, t, past + t)
        weights = torch.softmax(scores.masked_fill(~visible, float('-inf')), dim=-1)
        return self.proj((weights @ v).transpose(1, 2).reshape(b, t, width))


# %% [block] Transformer block
class Block(nn.Module):
    """Pre-LayerNorm: x + attention(LN(x)), then x + MLP(LN(x)) with a GELU and 4 × the width."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.attn = SelfAttention(c)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4 * c.width), nn.GELU(), nn.Linear(4 * c.width, c.width))

    def forward(self, x, cache: dict | None = None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


# %% [heads] Next frame token, reward and end
def head(width: int, classes: int):
    return nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, classes))


class Heads(nn.Module):
    """Three small MLPs on the last layer's output. Frame token i predicts frame token i + 1, and the action token
    predicts the next frame's first token, so every frame token is predicted from the tokens before it (the frame's
    last token has nothing to predict: the action is not modelled). The action token also predicts the reward
    (−1, 0 or +1) and whether the episode ends."""

    def __init__(self, c: Config):
        super().__init__()
        self.block, self.frame_tokens = c.block, c.frame_tokens
        self.observations = head(c.width, c.vocab)
        self.rewards = head(c.width, 3)
        self.ends = head(c.width, 2)

    def forward(self, x, start: int = 0):
        in_block = (start + torch.arange(x.shape[1], device=x.device)) % self.block
        next_frame_token = self.observations(x[:, in_block != self.frame_tokens - 1])
        actions = x[:, in_block == self.block - 1]
        return next_frame_token, self.rewards(actions), self.ends(actions)


class WorldModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.embedding = Embedding(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln_f = nn.LayerNorm(c.width)
        self.heads = Heads(c)

    def forward(self, tokens, cache: list | None = None):
        start = cache[0]['k'].shape[2] if cache and cache[0] else 0  # tokens already read
        x = self.embedding(tokens, start)
        for i, block in enumerate(self.blocks):
            x = block(x, None if cache is None else cache[i])
        return self.heads(self.ln_f(x), start)


# %% [imagine] Next-token sampling: one imagined step
@torch.no_grad()
def imagine_step(world_model, cache, action):
    """Feed the action; sample the reward and the end flag from its outputs; then sample the next frame's 16 tokens
    one at a time, each fed back in before the next is drawn. 17 calls, each reading one new token."""
    token, frame = action[:, None], []
    for k in range(17):
        next_token, reward, end = world_model(token, cache)
        if k == 0:
            reward = Categorical(logits=reward).sample().float() - 1  # classes 0, 1, 2 → −1, 0, +1
            end = Categorical(logits=end).sample().bool()
            first = (reward, end)
        if k < 16:
            token = Categorical(logits=next_token).sample()
            frame.append(token)
    return torch.cat(frame, dim=1), *first  # (b, 16) token ids, (b, 1) reward, (b, 1) end


# %% [actor-critic] Actor-critic
class ActorCritic(nn.Module):
    """Sees the decoded frame (never the real pixels): four 3 × 3 convolutions with max-pooling (64 → 4 px, 1024
    features), an LSTM cell that carries the past, and two linear heads: the policy's logits and the value."""

    def __init__(self, c: Config):
        super().__init__()
        widths = [3, 32, 32, 64, 64]
        self.convs = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(widths, widths[1:]))
        self.lstm = nn.LSTMCell(1024, c.lstm)
        self.critic = nn.Linear(c.lstm, 1)
        self.actor = nn.Linear(c.lstm, c.actions)

    def forward(self, frames, state):  # frames: (b, 3, 64, 64) in [0, 1]; state: (h, c) of the LSTM
        x = frames * 2 - 1
        for conv in self.convs:
            x = F.relu(F.max_pool2d(conv(x), 2))
        h, cell = self.lstm(x.flatten(1), state)
        return self.actor(h), self.critic(h)[:, 0], (h, cell)


# %% [returns] λ-returns and the losses
def lambda_returns(rewards, values, ends, c: Config):
    """G_t = r_t + γ·(1 − end_t)·((1 − λ)·V_{t+1} + λ·G_{t+1}), with G_T = V_T: a blend of every n-step return."""
    returns = values.clone()
    for t in reversed(range(rewards.shape[1] - 1)):
        bootstrap = (1 - c.lambda_) * values[:, t + 1] + c.lambda_ * returns[:, t + 1]
        returns[:, t] = rewards[:, t] + c.gamma * (~ends[:, t]) * bootstrap
    return returns


def losses(logits, values, actions, rewards, ends, c: Config):
    """REINFORCE with the value as baseline, the value regressed onto G, and a small entropy bonus. The last step
    only bootstraps."""
    returns = lambda_returns(rewards, values, ends, c)[:, :-1].detach()
    v = values[:, :-1]
    policy = Categorical(logits=logits[:, :-1])
    loss_actions = -(policy.log_prob(actions[:, :-1]) * (returns - v.detach())).mean()
    loss_values = F.mse_loss(v, returns)
    loss_entropy = -c.entropy_weight * policy.entropy().mean()
    return returns, loss_actions, loss_values, loss_entropy


# %% [model] Learning in imagination
@torch.no_grad()
def imagine(tokenizer, world_model, actor_critic, frames, actions, c: Config):
    """frames: (b, 21, 3, 64, 64) real frames, actions: (b, 20) the actions to feed. The actor-critic's LSTM is
    warmed up on the first 20 frames, each encoded and decoded; the world model starts from the last frame's tokens
    alone. Each step the actor-critic reads the current (decoded) frame, the action is fed and one step is imagined.
    The last step needs no next frame, only its reward and end."""
    b, horizon = actions.shape
    state = (frames.new_zeros(b, c.lstm), frames.new_zeros(b, c.lstm))
    for t in range(frames.shape[1] - 1):
        _, _, state = actor_critic(tokenizer.decode(tokenizer.encode(frames[:, t])), state)
    cache = [{} for _ in range(c.layers)]
    tokens = tokenizer.encode(frames[:, -1])
    world_model(tokens, cache)
    frame = tokenizer.decode(tokens)
    out = {'frames': [], 'logits': [], 'values': [], 'rewards': [], 'ends': []}
    for t in range(horizon):
        logits, value, state = actor_critic(frame, state)
        out['frames'].append(frame)
        out['logits'].append(logits)
        out['values'].append(value)
        if t < horizon - 1:
            tokens, reward, end = imagine_step(world_model, cache, actions[:, t])
            frame = tokenizer.decode(tokens)
        else:  # only the action's own pass
            _, reward, end = world_model(actions[:, t : t + 1], cache)
            reward = Categorical(logits=reward).sample().float() - 1
            end = Categorical(logits=end).sample().bool()
        out['rewards'].append(reward[:, 0])
        out['ends'].append(end[:, 0])
    out = {k: torch.stack(v, dim=1) for k, v in out.items()}
    out['returns'], *out['losses'] = losses(out['logits'], out['values'], actions, out['rewards'], out['ends'], c)
    return out
`;export{e as default};
