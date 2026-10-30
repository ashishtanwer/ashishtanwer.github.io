const e=`# Whisper base (Radford et al. 2022), openai/whisper-base, as plain PyTorch: how audio becomes the encoder's input
# and how the decoder's output becomes tokens. Audio at 16 kHz is padded with silence to 30 s and cut into 3,000
# frames of 80 log-mel energies (25 ms windows every 10 ms). The decoder writes byte-level BPE tokens after a prompt
# of special tokens: start of transcript, the language, the task, and either <|notimestamps|> or timestamp tokens,
# one per 20 ms (the spacing of the encoder's 1,500 positions), that bracket each segment of text.
# Each "# %% [id]" section is one step of the Whisper tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    sample_rate: int = 16_000
    n_samples: int = 480_000  # every clip is padded or cut to 30 s
    n_fft: int = 400  # 25 ms windows
    hop: int = 160  # one frame every 10 ms: 3,000 frames
    n_mels: int = 80
    eot: int = 50257  # <|endoftext|>
    sot: int = 50258  # <|startoftranscript|>
    # Language tokens follow <|startoftranscript|> in this order: <|en|> = 50259, <|zh|> = 50260, ...
    languages: str = (
        "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt "
        "la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc "
        "ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su"
    )
    translate: int = 50358
    transcribe: int = 50359
    no_timestamps: int = 50363
    timestamp_begin: int = 50364  # <|0.00|>; the last, 51864, is <|30.00|>
    seconds_per_timestamp: float = 0.02


# %% [stft] Pad to 30 s, then a power spectrum every 10 ms
def power_spectrum(audio, c: Config):
    """audio (samples,) at 16 kHz → (201 frequencies, 3,000 frames): |STFT|² with 400-sample Hann windows."""
    audio = F.pad(audio, (0, c.n_samples - audio.shape[-1]))  # pad with silence, or cut, to 30 s
    window = torch.hann_window(c.n_fft, device=audio.device)
    stft = torch.stft(audio, c.n_fft, c.hop, window=window, return_complex=True)
    return stft[..., :-1].abs() ** 2  # the last frame is dropped


# %% [mel] Mel filterbank: 201 frequencies → 80 bands
def mel_filters(c: Config):
    """Triangular filters, evenly spaced on Slaney's mel scale from 0 to 8 kHz, each with unit area: (201, 80)."""

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
    return (triangles * 2 / (edges[2:] - edges[:-2])).float()


# %% [log-mel] Log-mel spectrogram: the encoder's input
def log_mel(audio, c: Config):
    """audio (samples,) → (80, 3,000): log10 of each band's energy, floored 8 (80 dB) below the loudest cell of
    the whole 30 s, then scaled to roughly −1 … 1. The silence after the clip is all at that floor."""
    energy = mel_filters(c).to(audio.device).T @ power_spectrum(audio, c)
    log = energy.clamp(min=1e-10).log10()
    log = torch.maximum(log, log.max() - 8.0)
    return (log + 4.0) / 4.0


# %% [prompt] The decoder's prompt: special tokens
def prompt(c: Config, language="en", task="transcribe", timestamps=False):
    """The ids the decoder starts from: <|startoftranscript|> <|en|> <|transcribe|>, then <|notimestamps|> unless
    the text is to be bracketed by timestamp tokens."""
    ids = [c.sot, c.sot + 1 + c.languages.split().index(language), getattr(c, task)]
    return ids if timestamps else ids + [c.no_timestamps]


# %% [timestamps] Timestamp tokens: ids ↔ seconds, and the segments they bracket
def timestamp_id(seconds, c: Config):
    return c.timestamp_begin + round(seconds / c.seconds_per_timestamp)


def segments(ids, c: Config):
    """A timestamped output, e.g. <|0.00|> Where is my brother now?<|2.00|> → [(0.0, 2.0, [text ids])]: each
    segment opens and closes with a timestamp token; <|endoftext|> ends the output."""
    found, start, text = [], None, []
    for i in ids:
        if i == c.eot:
            break
        if i >= c.timestamp_begin:
            seconds = (i - c.timestamp_begin) * c.seconds_per_timestamp
            if start is None:
                start = seconds
            else:
                found.append((start, seconds, text))
                start, text = None, []
        elif start is not None:
            text.append(i)
    return found
`;export{e as default};
