const e=`# Mimi (Kyutai, Défossez et al. 2024), kyutai/mimi, as plain PyTorch: the split residual vector quantizer that
# turns the encoder's 12.5 Hz latent into codes. The encoder (convolutions with strides 4 · 5 · 6 · 8, a transformer,
# then a stride-2 convolution) turns 24,000 samples a second into 12.5 frames of 512 numbers. Two quantizers read
# that same latent: a semantic one with a single codebook, which training distilled toward WavLM's speech features,
# and an acoustic one whose codebooks quantize residuals. Each projects 512 → 256, searches codebooks of 2,048
# vectors (11 bits) and projects back; the two results are added. Moshi keeps 8 codes per frame: 1.1 kbps.
# Each "# %% [id]" section is one step of the Mimi tab's diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Config:
    sampling_rate: int = 24000
    frame_rate: float = 12.5  # 1,920 samples per frame
    hidden: int = 512  # latent width
    dim: int = 256  # quantizer width
    codebook_size: int = 2048  # 11 bits per code
    semantic: int = 1  # codebooks in the semantic quantizer
    acoustic: int = 31  # codebooks in the acoustic quantizer; the checkpoint has 32 in all
    used: int = 8  # codes per frame Moshi keeps: 1 semantic + 7 acoustic


class RVQ(nn.Module):
    """One residual vector quantizer's weights: the 1 × 1 projections in and out, and its codebooks. The released
    model stores each codebook as a running sum and a usage count; \`codebooks\` holds their ratio, the vectors the
    model searches."""

    def __init__(self, c: Config, codebooks: int):
        super().__init__()
        self.project_in = nn.Linear(c.hidden, c.dim, bias=False)
        self.project_out = nn.Linear(c.dim, c.hidden, bias=False)
        self.register_buffer("codebooks", torch.zeros(codebooks, c.codebook_size, c.dim))


class SplitRVQ(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.semantic = RVQ(c, c.semantic)
        self.acoustic = RVQ(c, c.acoustic)


# %% [nearest] One codebook: the nearest vector
def nearest(x, codebook):
    """x (frames, dim), codebook (size, dim) → the id of the nearest codebook vector for each frame, (frames,)."""
    return torch.cdist(x[None], codebook[None])[0].argmin(-1)


# %% [rvq-encode] One quantizer: project, then quantize residuals
def rvq_encode(latent, rvq: RVQ, n):
    """latent (frames, 512) → codes (n, frames). Each codebook quantizes what the earlier ones left over."""
    residual = rvq.project_in(latent)  # (frames, 256)
    codes = []
    for codebook in rvq.codebooks[:n]:
        ids = nearest(residual, codebook)
        residual = residual - codebook[ids]
        codes.append(ids)
    return torch.stack(codes)


# %% [split-encode] Split RVQ: semantic and acoustic codes from the same latent
def split_encode(latent, q: SplitRVQ, n=8):
    """latent (frames, 512) → codes (n, frames): code 1 from the semantic quantizer, codes 2..n from the acoustic
    one. The acoustic quantizer reads the latent itself, not what the semantic code left over."""
    codes = rvq_encode(latent, q.semantic, q.semantic.codebooks.shape[0])
    if n > codes.shape[0]:
        codes = torch.cat([codes, rvq_encode(latent, q.acoustic, n - codes.shape[0])])
    return codes


# %% [rvq-decode] One quantizer: sum the chosen vectors, project back
def rvq_decode(codes, rvq: RVQ):
    """codes (k, frames) → (frames, 512)."""
    quantized = sum(codebook[ids] for codebook, ids in zip(rvq.codebooks, codes))
    return rvq.project_out(quantized)


# %% [split-decode] Split RVQ: codes → the quantized latent
def split_decode(codes, q: SplitRVQ):
    """codes (k, frames) → (frames, 512): the semantic part plus the acoustic part. With k = 1 only the semantic
    code is decoded; the decoder turns either back into a waveform."""
    k = q.semantic.codebooks.shape[0]
    latent = rvq_decode(codes[:k], q.semantic)
    if codes.shape[0] > k:
        latent = latent + rvq_decode(codes[k:], q.acoustic)
    return latent
`;export{e as default};
