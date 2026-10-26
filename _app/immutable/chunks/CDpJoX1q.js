const e=`# Whisper base, the model in the Speech tab, written as plain PyTorch blocks.
# Each "# %% [id]" section is one block of the diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    sample_rate: int = 16_000
    n_samples: int = 480_000  # every clip is padded or cut to 30 s
    n_fft: int = 400  # 25 ms windows
    hop: int = 160  # one frame every 10 ms: 3,000 frames
    n_mels: int = 80
    vocab_size: int = 51865
    d_model: int = 512
    n_layers: int = 6  # in each stack
    n_heads: int = 8  # 64 dimensions per head
    d_mlp: int = 2048
    n_audio_positions: int = 1500  # the stride-2 convolution halves 3,000 frames: 20 ms each
    n_text_positions: int = 448
    eps: float = 1e-5


# %% [log-mel] Log-mel spectrogram
def mel_filters(c: Config):
    """Triangular filters, evenly spaced on Slaney's mel scale from 0 to 8 kHz, each with unit area."""

    def to_mel(hz):  # linear below 1 kHz, logarithmic above
        return torch.where(hz < 1000, 3 * hz / 200, 15 + torch.log(hz / 1000) * 27 / math.log(6.4))

    def to_hz(mel):
        return torch.where(mel < 15, 200 * mel / 3, 1000 * torch.exp((mel - 15) * math.log(6.4) / 27))

    low, high = to_mel(torch.tensor([0.0, c.sample_rate / 2], dtype=torch.float64))
    # Filter i rises from edge i to edge i + 1 and falls to edge i + 2.
    edges = to_hz(torch.linspace(low, high, c.n_mels + 2, dtype=torch.float64))
    frequencies = torch.linspace(0, c.sample_rate // 2, c.n_fft // 2 + 1, dtype=torch.float64)
    slopes = edges[None, :] - frequencies[:, None]  # (201, n_mels + 2)
    rising = -slopes[:, :-2] / edges.diff()[:-1]
    falling = slopes[:, 2:] / edges.diff()[1:]
    triangles = torch.minimum(rising, falling).clamp(min=0)
    return (triangles * 2 / (edges[2:] - edges[:-2])).float()  # (201, n_mels)


def log_mel(audio, c: Config):  # audio: (samples,) at 16 kHz
    audio = F.pad(audio, (0, c.n_samples - audio.shape[-1]))  # pad with silence, or cut, to 30 s
    window = torch.hann_window(c.n_fft, device=audio.device)
    stft = torch.stft(audio, c.n_fft, c.hop, window=window, return_complex=True)
    power = stft[..., :-1].abs() ** 2  # (201 frequencies, 3,000 frames): the last frame is dropped
    log = (mel_filters(c).to(audio.device).T @ power).clamp(min=1e-10).log10()  # (n_mels, 3,000)
    log = torch.maximum(log, log.max() - 8.0)  # keep the top 80 dB
    return (log + 4.0) / 4.0


# %% [conv-frontend] Conv front end
def sinusoids(length, channels, max_timescale=10_000):
    """Fixed position vectors: sines in the first half of the channels, cosines in the second."""
    rates = torch.exp(-math.log(max_timescale) / (channels // 2 - 1) * torch.arange(channels // 2))
    angles = torch.arange(length)[:, None] * rates[None, :]  # (length, channels / 2)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ConvFrontEnd(nn.Module):
    """Two convolutions over time, each followed by GELU; the second has stride 2. Then the fixed
    sinusoidal position vectors are added. This is the encoder's embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.conv1 = nn.Conv1d(c.n_mels, c.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c.d_model, c.d_model, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positions", sinusoids(c.n_audio_positions, c.d_model))

    def forward(self, mel):  # mel: (batch, n_mels, 3,000)
        x = F.gelu(self.conv1(mel))  # (batch, d_model, 3,000)
        x = F.gelu(self.conv2(x))  # (batch, d_model, 1,500)
        return x.transpose(1, 2) + self.positions  # (batch, 1,500, d_model)


# %% [attention] Multi-head self-attention
class Attention(nn.Module):
    """Query, key, value and output projections; only the key projection has no bias."""

    def __init__(self, c: Config):
        super().__init__()
        self.n_heads = c.n_heads
        self.q = nn.Linear(c.d_model, c.d_model)
        self.k = nn.Linear(c.d_model, c.d_model, bias=False)
        self.v = nn.Linear(c.d_model, c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)

    def heads(self, z):  # (batch, seq, d_model) → (batch, heads, seq, head_dim)
        return z.view(z.shape[0], z.shape[1], self.n_heads, -1).transpose(1, 2)

    def attend(self, x, memory, causal):
        q, k, v = self.heads(self.q(x)), self.heads(self.k(memory)), self.heads(self.v(memory))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (batch, heads, queries, keys)
        if causal:
            future = torch.ones(scores.shape[-2:], dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(future, float("-inf"))
        weights = scores.softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(x.shape))

    def forward(self, x, causal=False):  # the decoder passes causal=True: no peeking at later tokens
        return self.attend(x, x, causal)


# %% [mlp] MLP
class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.d_model, c.d_mlp)  # fc1
        self.down = nn.Linear(c.d_mlp, c.d_model)  # fc2

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


# %% [encoder-block] Encoder block
class EncoderBlock(nn.Module):
    """Pre-norm: x + SelfAttention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# %% [encoder] Encoder hidden state
class Encoder(nn.Module):
    """The conv front end, the encoder blocks, then a final LayerNorm. Its output is read by every
    decoder block's cross-attention."""

    def __init__(self, c: Config):
        super().__init__()
        self.frontend = ConvFrontEnd(c)
        self.blocks = nn.ModuleList(EncoderBlock(c) for _ in range(c.n_layers))
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)

    def forward(self, mel):
        x = self.frontend(mel)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)  # (batch, 1,500, d_model)


# %% [embedding] Decoder embedding
class Embedding(nn.Module):
    """Token embedding plus a learned position embedding."""

    def __init__(self, c: Config):
        super().__init__()
        self.token = nn.Embedding(c.vocab_size, c.d_model)
        self.position = nn.Embedding(c.n_text_positions, c.d_model)

    def forward(self, ids):  # ids: (batch, tokens)
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.token(ids) + self.position(positions)  # (batch, tokens, d_model)


# %% [cross-attention] Cross-attention
class CrossAttention(Attention):
    """Queries come from the decoder, keys and values from the encoder hidden state. No mask: every
    token sees all 1,500 audio positions."""

    def forward(self, x, encoder_out):  # x: (batch, tokens, d), encoder_out: (batch, 1,500, d)
        return self.attend(x, encoder_out, causal=False)


# %% [decoder-block] Decoder block
class DecoderBlock(nn.Module):
    """Masked self-attention, cross-attention, then the MLP, each pre-norm with a residual."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm1 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.attention = Attention(c)
        self.norm2 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.cross_attention = CrossAttention(c)
        self.norm3 = nn.LayerNorm(c.d_model, eps=c.eps)
        self.mlp = MLP(c)

    def forward(self, x, encoder_out):
        x = x + self.attention(self.norm1(x), causal=True)
        x = x + self.cross_attention(self.norm2(x), encoder_out)
        return x + self.mlp(self.norm3(x))


# %% [output] Probabilities
class Output(nn.Module):
    """Final LayerNorm, then the token embedding reused as the output projection, without scaling."""

    def __init__(self, c: Config):
        super().__init__()
        self.norm = nn.LayerNorm(c.d_model, eps=c.eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)  # shares the token embedding

    def forward(self, x):
        return self.lm_head(self.norm(x))  # logits: (batch, tokens, vocab)


def next_token_probabilities(logits, temperature=1.0, top_k=None, top_p=None):
    """The last position's logits, scaled by temperature and cut to top-k or top-p, as probabilities."""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        p = sorted_logits.softmax(dim=-1)
        drop = p.cumsum(dim=-1) - p >= top_p  # keep the smallest set whose mass reaches top_p
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))
    return logits.softmax(dim=-1)


# %% [model] Full model
class Whisper(nn.Module):
    def __init__(self, c: Config = Config()):
        super().__init__()
        self.encoder = Encoder(c)
        self.embedding = Embedding(c)
        self.decoder = nn.ModuleList(DecoderBlock(c) for _ in range(c.n_layers))
        self.output = Output(c)
        self.output.lm_head.weight = self.embedding.token.weight  # weight tying

    def forward(self, mel, ids):  # mel: (batch, n_mels, 3,000) from log_mel; ids: (batch, tokens)
        encoder_out = self.encoder(mel)
        x = self.embedding(ids)
        for block in self.decoder:
            x = block(x, encoder_out)
        return self.output(x)


# mel = log_mel(audio, Config())[None]
# ids = torch.tensor([[50258, 50259, 50359, 50363]])  # start, English, transcribe, no timestamps
# probabilities = next_token_probabilities(Whisper()(mel, ids), top_k=5)
`;export{e as default};
