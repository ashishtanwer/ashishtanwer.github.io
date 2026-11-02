const e=`# Kokoro-82M (hexgrad, 2025), hexgrad/Kokoro-82M, as plain PyTorch: StyleTTS 2's inference path (Li et al. 2023)
# with an iSTFTNet vocoder (Kaneko et al. 2022). Text becomes phonemes outside the network (misaki's G2P); the
# network reads one id per phoneme (178 symbols) and a 256-number style vector taken from a voice pack, and writes
# 24 kHz audio. A PL-BERT encoder (ALBERT) and a duration predictor decide how many 25 ms frames each phoneme lasts;
# a prosody predictor draws pitch (F0) and energy at 80 Hz; a decoder whose every normalisation is modulated by the
# style turns phoneme features, pitch and energy into the magnitude and phase of a 20-point STFT, and the inverse
# STFT turns those into samples. Nothing is sampled except the vocoder's harmonic source: random start phases for
# its overtones, and a little noise.
# Each "# %% [id]" section is one step of the Kokoro tab's diagram.
import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    n_token: int = 178  # phoneme symbols; id 0 marks the start and the end
    style_dim: int = 128  # each half of the 256-number style vector
    hidden: int = 512
    layers: int = 3  # LSTM + AdaLayerNorm pairs in the duration encoder, convolutions in the text encoder
    max_dur: int = 50  # duration bins: a phoneme lasts at most 50 frames
    bert_embedding: int = 128  # ALBERT's factorised embedding, mapped up to bert_hidden
    bert_hidden: int = 768
    bert_heads: int = 12
    bert_ffn: int = 2048
    bert_layers: int = 12  # one shared layer, applied 12 times
    bert_positions: int = 512
    decoder_width: int = 1024
    asr_res: int = 64  # phoneme features re-added to every decoder block
    upsample_rates: tuple = (10, 6)
    upsample_kernels: tuple = (20, 12)
    upsample_channels: int = 512
    resblock_kernels: tuple = (3, 7, 11)
    resblock_dilations: tuple = (1, 3, 5)
    n_fft: int = 20  # 11 frequency bins
    hop: int = 5  # one STFT frame per 5 samples; 600 samples (25 ms) per duration frame
    sampling_rate: int = 24000
    harmonics: int = 8  # overtones above F0 in the harmonic source
    voiced_threshold: float = 10.0  # Hz
    sine_amp: float = 0.1
    noise_std: float = 0.003


# %% [checkpoint] The released weights
def load_release(checkpoint):
    """kokoro-v1_0.pth (one state dict per part, keys prefixed "module.") → this file's state dict. Convolutions
    were trained with weight normalisation: weight = g · v / ‖v‖, the norm over every axis but the first. ALBERT's
    pooler is in the file but nothing reads it."""
    state = {}
    for part, weights in checkpoint.items():
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
        for key, value in weights.items():
            if key.endswith("weight_v") or key.startswith("pooler."):
                continue
            if key.endswith("weight_g"):  # g · v / ‖v‖ with torch's own kernel, so it is the trained weight to the bit
                key, value = key[:-2], torch._weight_norm(weights[key[:-1] + "v"], value, 0)
            state[f"{part}.{rename(part, key)}"] = value
    return state


def rename(part, key):
    """A released key → this file's name for the same weight."""
    rules = {
        "bert": [
            (r"^embeddings\\.word_embeddings", "word"),
            (r"^embeddings\\.position_embeddings", "position"),
            (r"^embeddings\\.token_type_embeddings", "token_type"),
            (r"^embeddings\\.LayerNorm", "embed_norm"),
            (r"^encoder\\.embedding_hidden_mapping_in", "embed_in"),
            (r"^encoder\\.albert_layer_groups\\.0\\.albert_layers\\.0\\.", ""),
            (r"^attention\\.query", "q"),
            (r"^attention\\.key", "k"),
            (r"^attention\\.value", "v"),
            (r"^attention\\.dense", "o"),
            (r"^attention\\.LayerNorm", "attn_norm"),
            (r"^ffn_output", "ffn_out"),
            (r"^full_layer_layer_norm", "ffn_norm"),
        ],
        "predictor": [
            (r"^text_encoder\\.lstms\\.(\\d+)\\.", lambda m: f"encoder.{int(m[1]) // 2}.{('lstm', 'norm')[int(m[1]) % 2]}."),
            (r"^duration_proj\\.linear_layer", "duration_proj"),
        ],
        "text_encoder": [
            (r"^cnn\\.(\\d)\\.0\\.", r"convs.\\1."),
            (r"^cnn\\.(\\d)\\.1\\.gamma", r"norms.\\1.weight"),
            (r"^cnn\\.(\\d)\\.1\\.beta", r"norms.\\1.bias"),
        ],
        "decoder": [(r"^asr_res\\.0\\.", "asr_res."), (r"^generator\\.m_source\\.l_linear", "generator.source_merge")],
    }
    for pattern, replacement in rules.get(part, []):
        key = re.sub(pattern, replacement, key)
    return key


# %% [style] The style vector
def style_vector(pack, phonemes: int, c: Config):
    """A voice pack holds one 256-number style per utterance length, (510, 1, 256); Kokoro takes the row for the
    number of phonemes. The first half conditions the decoder (timbre), the second the predictors (prosody)."""
    style = pack[phonemes - 1]  # (1, 256)
    return style[:, : c.style_dim], style[:, c.style_dim :]  # (decoder, predictor), each (1, 128)


# %% [plbert] PL-BERT: phonemes in context
class PLBert(nn.Module):
    """ALBERT pre-trained on phonemes: a 128-number embedding mapped to 768, then one layer applied 12 times."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.word = nn.Embedding(c.n_token, c.bert_embedding)
        self.position = nn.Embedding(c.bert_positions, c.bert_embedding)
        self.token_type = nn.Embedding(2, c.bert_embedding)
        self.embed_norm = nn.LayerNorm(c.bert_embedding, eps=1e-12)
        self.embed_in = nn.Linear(c.bert_embedding, c.bert_hidden)
        self.q, self.k, self.v, self.o = (nn.Linear(c.bert_hidden, c.bert_hidden) for _ in range(4))
        self.attn_norm = nn.LayerNorm(c.bert_hidden, eps=1e-12)
        self.ffn = nn.Linear(c.bert_hidden, c.bert_ffn)
        self.ffn_out = nn.Linear(c.bert_ffn, c.bert_hidden)
        self.ffn_norm = nn.LayerNorm(c.bert_hidden, eps=1e-12)

    def forward(self, ids):  # ids (1, T) → (1, T, 768)
        T, H = ids.shape[1], self.c.bert_heads
        x = self.word(ids) + self.token_type.weight[0] + self.position(torch.arange(T))
        x = self.embed_in(self.embed_norm(x))
        for _ in range(self.c.bert_layers):  # the same weights every time
            q, k, v = (proj(x).view(1, T, H, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
            attended = F.scaled_dot_product_attention(q, k, v)  # softmax(q·kᵀ / √d) · v
            x = self.attn_norm(x + self.o(attended.transpose(1, 2).reshape(1, T, -1)))
            h = self.ffn(x)  # GELU, tanh form
            h = 0.5 * h * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * torch.pow(h, 3.0))))
            x = self.ffn_norm(x + self.ffn_out(h))
        return x


# %% [adain] Adaptive instance normalisation
class AdaIN(nn.Module):
    """Normalise each channel over time, then scale and shift it by amounts read from the style."""

    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, 2 * channels)

    def forward(self, x, s):  # x (1, C, T), s (1, style)
        gamma, beta = self.fc(s)[:, :, None].chunk(2, dim=1)
        return (1 + gamma) * F.instance_norm(x, eps=1e-5) + beta


class AdaLayerNorm(nn.Module):
    """The predictor's version: layer norm over channels, per frame, with a style scale and shift."""

    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, 2 * channels)

    def forward(self, x, s):  # x (1, T, C)
        gamma, beta = self.fc(s)[:, None].chunk(2, dim=-1)
        return (1 + gamma) * F.layer_norm(x, x.shape[-1:], eps=1e-5) + beta


class AdaINBlock(nn.Module):
    """StyleTTS 2's residual block: AdaIN → LeakyReLU → conv, twice; with upsample, a depthwise transposed
    convolution doubles the frames and the shortcut repeats each frame. The sum is divided by √2."""

    def __init__(self, dim_in, dim_out, style_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.norm1, self.norm2 = AdaIN(style_dim, dim_in), AdaIN(style_dim, dim_out)
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, padding=1)
        self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, bias=False) if dim_in != dim_out else None
        if upsample:
            self.pool = nn.ConvTranspose1d(dim_in, dim_in, 3, 2, padding=1, output_padding=1, groups=dim_in)

    def forward(self, x, s):
        shortcut = F.interpolate(x, scale_factor=2, mode="nearest") if self.upsample else x
        if self.conv1x1 is not None:
            shortcut = self.conv1x1(shortcut)
        h = F.leaky_relu(self.norm1(x, s), 0.2)
        if self.upsample:
            h = self.pool(h)
        h = self.conv2(F.leaky_relu(self.norm2(self.conv1(h), s), 0.2))
        return (h + shortcut) * torch.rsqrt(torch.tensor(2.0))


# %% [durations] How long each phoneme lasts
class Predictor(nn.Module):
    """Reads PL-BERT's features with the prosody half of the style: a duration encoder (BiLSTM + AdaLayerNorm, 3
    times), then a BiLSTM and a projection to 50 bins whose sigmoids add up to the phoneme's frames."""

    def __init__(self, c: Config):
        super().__init__()
        d, s = c.hidden, c.style_dim
        self.encoder = nn.ModuleList(
            nn.ModuleDict({"lstm": nn.LSTM(d + s, d // 2, batch_first=True, bidirectional=True),
                           "norm": AdaLayerNorm(s, d)})
            for _ in range(c.layers)
        )
        self.lstm = nn.LSTM(d + s, d // 2, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(d, c.max_dur)
        self.shared = nn.LSTM(d + s, d // 2, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([AdaINBlock(d, d, s), AdaINBlock(d, d // 2, s, True), AdaINBlock(d // 2, d // 2, s)])
        self.N = nn.ModuleList([AdaINBlock(d, d, s), AdaINBlock(d, d // 2, s, True), AdaINBlock(d // 2, d // 2, s)])
        self.F0_proj = nn.Conv1d(d // 2, 1, 1)
        self.N_proj = nn.Conv1d(d // 2, 1, 1)

    def encode(self, features, s):  # features (1, T, 512) → (1, T, 640): each step's output with the style
        style = s[:, None].expand(-1, features.shape[1], -1)
        x = torch.cat([features, style], -1)
        for layer in self.encoder:
            x = torch.cat([layer["norm"](layer["lstm"](x)[0], s), style], -1)
        return x


def durations(predictor: Predictor, d, speed=1.0):
    """d (1, T, 640) → the 50 duration logits (1, T, 50) and whole frames per phoneme (T,), at least 1."""
    logits = predictor.duration_proj(predictor.lstm(d)[0])
    frames = torch.sigmoid(logits).sum(-1)[0] / speed
    return logits, torch.round(frames).clamp(min=1).long()


# %% [alignment] Phonemes spread over frames
def alignment(frames):
    """frames (T,) → a 0/1 matrix (T, sum(frames)): frame j belongs to the phoneme whose span holds it."""
    owner = torch.repeat_interleave(torch.arange(frames.shape[0]), frames)
    return F.one_hot(owner, frames.shape[0]).T.float()


# %% [prosody] Pitch and energy
def prosody(predictor: Predictor, d, align, s):
    """The encoder's output stretched to frames, d (1, T, 640) and align (T, F) → F0 in Hz and energy, each
    (1, 2F): the middle block upsamples, so both curves have two values per 25 ms frame."""
    x = predictor.shared((d.transpose(1, 2) @ align).transpose(1, 2))[0].transpose(1, 2)  # (1, 512, F)
    f0, n = x, x
    for block in predictor.F0:
        f0 = block(f0, s)
    for block in predictor.N:
        n = block(n, s)
    return predictor.F0_proj(f0)[:, 0], predictor.N_proj(n)[:, 0]


# %% [text-encoder] Phoneme features for the decoder
class TextEncoder(nn.Module):
    """An embedding, three convolutions (kernel 5, layer norm, LeakyReLU) and a BiLSTM: (1, T) → (1, 512, T)."""

    def __init__(self, c: Config):
        super().__init__()
        self.embedding = nn.Embedding(c.n_token, c.hidden)
        self.convs = nn.ModuleList(nn.Conv1d(c.hidden, c.hidden, 5, padding=2) for _ in range(c.layers))
        self.norms = nn.ModuleList(nn.LayerNorm(c.hidden, eps=1e-5) for _ in range(c.layers))
        self.lstm = nn.LSTM(c.hidden, c.hidden // 2, batch_first=True, bidirectional=True)

    def forward(self, ids):
        x = self.embedding(ids).transpose(1, 2)
        for conv, norm in zip(self.convs, self.norms):
            x = F.leaky_relu(norm(conv(x).transpose(1, 2)).transpose(1, 2), 0.2)
        return self.lstm(x.transpose(1, 2))[0].transpose(1, 2)


# %% [decoder] The decoder
class Decoder(nn.Module):
    """Phoneme features at frame rate with F0 and energy → 512 channels at 80 Hz for the vocoder. Every block
    sees the timbre half of the style, and is fed the features, F0 and energy again."""

    def __init__(self, c: Config):
        super().__init__()
        w, s, extra = c.decoder_width, c.style_dim, c.asr_res + 2
        self.F0_conv = nn.Conv1d(1, 1, 3, stride=2, padding=1)  # 80 Hz → 40 Hz
        self.N_conv = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        self.encode = AdaINBlock(c.hidden + 2, w, s)
        self.asr_res = nn.Conv1d(c.hidden, c.asr_res, 1)
        self.decode = nn.ModuleList([AdaINBlock(w + extra, w, s) for _ in range(3)]
                                    + [AdaINBlock(w + extra, c.upsample_channels, s, upsample=True)])
        self.generator = Generator(c)

    def features(self, asr, f0, n, s):  # asr (1, 512, F), f0 and n (1, 2F) → (1, 512, 2F)
        f0, n = self.F0_conv(f0[:, None]), self.N_conv(n[:, None])
        x = self.encode(torch.cat([asr, f0, n], 1), s)
        residual = self.asr_res(asr)
        for block in self.decode:
            x = block(torch.cat([x, residual, f0, n], 1), s)
        return x


# %% [source] The harmonic source
def harmonic_source(f0, c: Config, merge: nn.Linear):
    """F0 (1, 2F) → a 24 kHz excitation (1, 600F): sines at F0 and 8 overtones (random start phases for the
    overtones), silent where F0 ≤ 10 Hz, plus noise, merged by a 9 → 1 linear layer and tanh. The phase is summed
    at the frame rate and interpolated back, so it stays smooth."""
    up = math.prod(c.upsample_rates) * c.hop  # samples per F0 value: 300
    f0 = F.interpolate(f0[:, None], scale_factor=up, mode="nearest").transpose(1, 2)  # (1, L, 1)
    harmonics = f0 * torch.arange(1, c.harmonics + 2)  # (1, L, 9)
    rad = (harmonics / c.sampling_rate) % 1  # cycles per sample
    start = torch.rand(1, c.harmonics + 1)
    start[:, 0] = 0
    rad[:, 0] = rad[:, 0] + start
    rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / up, mode="linear").transpose(1, 2)
    phase = torch.cumsum(rad, dim=1) * 2 * torch.pi
    phase = F.interpolate(phase.transpose(1, 2) * up, scale_factor=up, mode="linear").transpose(1, 2)
    voiced = (f0 > c.voiced_threshold).float()
    noise = (voiced * c.noise_std + (1 - voiced) * c.sine_amp / 3) * torch.randn_like(phase)
    sines = c.sine_amp * torch.sin(phase) * voiced + noise
    return torch.tanh(merge(sines))[:, :, 0]


# %% [generator] iSTFTNet: upsample to STFT frames
class SnakeBlock(nn.Module):
    """Three dilated convolution pairs, each after AdaIN and a Snake activation x + sin²(αx)/α."""

    def __init__(self, channels, kernel, dilations, style_dim):
        super().__init__()
        pad = lambda d: (kernel * d - d) // 2  # noqa: E731
        self.convs1 = nn.ModuleList(nn.Conv1d(channels, channels, kernel, dilation=d, padding=pad(d)) for d in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(channels, channels, kernel, padding=pad(1)) for _ in dilations)
        self.adain1 = nn.ModuleList(AdaIN(style_dim, channels) for _ in dilations)
        self.adain2 = nn.ModuleList(AdaIN(style_dim, channels) for _ in dilations)
        self.alpha1 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)
        self.alpha2 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            h = n1(x, s)
            h = c1(h + torch.sin(a1 * h) ** 2 / a1)
            h = n2(h, s)
            x = x + c2(h + torch.sin(a2 * h) ** 2 / a2)
        return x


class Generator(nn.Module):
    """512 channels at 80 Hz → 22 channels at 4.8 kHz (one per STFT frame): two transposed convolutions (×10, ×6),
    each followed by the mean of three Snake blocks; the harmonic source's own STFT is added at each scale."""

    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        bins2 = c.n_fft + 2  # magnitude and phase of 11 bins
        self.source_merge = nn.Linear(c.harmonics + 1, 1)
        self.ups, self.noise_convs, self.noise_res, self.resblocks = (nn.ModuleList() for _ in range(4))
        for i, (rate, kernel) in enumerate(zip(c.upsample_rates, c.upsample_kernels)):
            ch = c.upsample_channels // 2 ** (i + 1)
            self.ups.append(nn.ConvTranspose1d(2 * ch, ch, kernel, rate, padding=(kernel - rate) // 2))
            self.resblocks.extend(SnakeBlock(ch, k, c.resblock_dilations, c.style_dim) for k in c.resblock_kernels)
            later = math.prod(c.upsample_rates[i + 1 :])
            if i + 1 < len(c.upsample_rates):
                self.noise_convs.append(nn.Conv1d(bins2, ch, later * 2, stride=later, padding=(later + 1) // 2))
                self.noise_res.append(SnakeBlock(ch, 7, (1, 3, 5), c.style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(bins2, ch, 1))
                self.noise_res.append(SnakeBlock(ch, 11, (1, 3, 5), c.style_dim))
        self.conv_post = nn.Conv1d(ch, bins2, 7, padding=3)

    def forward(self, x, s, f0):
        """x (1, 512, 2F), f0 (1, 2F) → log-magnitude and phase logits, each (1, 11, 120F + 1)."""
        c = self.c
        excitation = harmonic_source(f0, c, self.source_merge)
        window = torch.hann_window(c.n_fft)
        spec = torch.stft(excitation, c.n_fft, c.hop, window=window, return_complex=True)
        source = torch.cat([spec.abs(), spec.angle()], 1)  # (1, 22, 120F + 1)
        k = len(c.resblock_kernels)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            if i == len(self.ups) - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            x = x + self.noise_res[i](self.noise_convs[i](source), s)
            x = sum(block(x, s) for block in self.resblocks[i * k : (i + 1) * k]) / k
        out = self.conv_post(F.leaky_relu(x))
        return out[:, : c.n_fft // 2 + 1], out[:, c.n_fft // 2 + 1 :]


# %% [istft] Magnitude and phase → samples
def istft(log_magnitude, phase_logits, c: Config):
    """(1, 11, N) each → audio (1, 5(N − 1)). The magnitude is exp(·) and the phase sin(·), so every phase lies in
    [−1, 1] radians. Each frame's 20 samples are the inverse real FFT of its 11 bins, windowed (Hann) and
    overlap-added every 5 samples, divided by the summed squared window; 10 samples are cut from each end."""
    magnitude, phase = torch.exp(log_magnitude), torch.sin(phase_logits)
    window = torch.hann_window(c.n_fft)
    frames = torch.fft.irfft(torch.polar(magnitude, phase), n=c.n_fft, dim=1) * window[:, None]  # (1, 20, N)
    N = frames.shape[-1]
    length = c.n_fft + c.hop * (N - 1)
    audio = F.fold(frames, (1, length), (1, c.n_fft), stride=(1, c.hop))[:, 0, 0]
    envelope = F.fold(window.square()[None, :, None].expand(1, -1, N), (1, length), (1, c.n_fft), stride=(1, c.hop))
    return (audio / envelope[:, 0, 0])[:, c.n_fft // 2 : length - c.n_fft // 2]


# %% [model] Phonemes and a style → audio
class Kokoro(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.bert = PLBert(c)
        self.bert_encoder = nn.Linear(c.bert_hidden, c.hidden)
        self.predictor = Predictor(c)
        self.text_encoder = TextEncoder(c)
        self.decoder = Decoder(c)

    def forward(self, ids, style, speed=1.0):
        """ids (1, T) with 0 at both ends, style (1, 256) → audio (1, 600F) and frames per phoneme (T,)."""
        s_decoder, s_prosody = style[:, : self.c.style_dim], style[:, self.c.style_dim :]
        d = self.predictor.encode(self.bert_encoder(self.bert(ids)), s_prosody)
        _, frames = durations(self.predictor, d, speed)
        align = alignment(frames)
        f0, n = prosody(self.predictor, d, align, s_prosody)
        asr = self.text_encoder(ids) @ align  # (1, 512, F)
        x = self.decoder.features(asr, f0, n, s_decoder)
        return istft(*self.decoder.generator(x, s_decoder, f0), self.c), frames
`;export{e as default};
