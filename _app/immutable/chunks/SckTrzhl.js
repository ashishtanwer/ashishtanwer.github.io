const e=`# EnCodec 24 kHz (Défossez et al. 2022), facebook/encodec_24khz, as plain PyTorch: the residual vector quantizer
# that turns the encoder's latent into codes. The convolutional encoder turns 24,000 samples a second into 75 frames
# of 128 numbers (strides 2 · 4 · 5 · 8 = 320 samples per frame). Each of 32 codebooks holds 1,024 vectors of 128:
# the first replaces a frame by its nearest vector, the second the nearest to what the first left over, and so on.
# A frame's tokens are the 32 chosen ids, 10 bits each; keeping the first k codebooks gives 0.75 · k kbps.
# \`codebooks\` below is the released model's (32, 1024, 128) tensor: codebook q, vector i is codebooks[q, i].
# Each "# %% [id]" section is one step of the EnCodec tab's diagram.
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    sampling_rate: int = 24000
    hop: int = 320  # samples per frame: the encoder's strides 2 · 4 · 5 · 8
    dim: int = 128  # latent and codebook vector width
    codebook_size: int = 1024  # 10 bits per code
    codebooks: int = 32  # 32 · 10 bits · 75 frames/s = 24 kbps
    bandwidths: tuple = (1.5, 3.0, 6.0, 12.0, 24.0)  # kbps the model was trained for

    @property
    def frame_rate(self):
        return self.sampling_rate / self.hop  # 75 frames per second


# %% [nearest] One codebook: the nearest vector
def nearest(x, codebook):
    """x (frames, dim), codebook (size, dim) → the id of the nearest codebook vector for each frame, (frames,).
    |x − e|² = |x|² − 2 x·e + |e|², written out as the model computes it."""
    distance = x.pow(2).sum(-1, keepdim=True) - 2 * x @ codebook.T + codebook.pow(2).sum(-1)
    return distance.argmin(-1)


# %% [rvq-encode] Residual vector quantization: latent → codes
def rvq_encode(latent, codebooks, n=None):
    """latent (dim, frames) → codes (n, frames). Each codebook quantizes what the earlier ones left over, so the
    first k codes of a frame are the same whether 2 or 32 codebooks are kept."""
    residual = latent.T  # (frames, dim)
    codes = []
    for codebook in codebooks[:n]:
        ids = nearest(residual, codebook)
        residual = residual - codebook[ids]  # what is left for the next codebook
        codes.append(ids)
    return torch.stack(codes)


# %% [rvq-decode] Decoding: codes → the quantized latent
def rvq_decode(codes, codebooks):
    """codes (k, frames) → (dim, frames): the sum of each codebook's chosen vector. With fewer codebooks the sum
    stops earlier and the latent is coarser; the decoder turns either back into a waveform."""
    quantized = sum(codebook[ids] for codebook, ids in zip(codebooks, codes))
    return quantized.T


# %% [bandwidth] Bandwidth → number of codebooks
def codebooks_for(kbps, c: Config):
    """Each codebook adds log2(1024) = 10 bits per frame, 750 bits per second at 75 frames per second."""
    per_codebook = math.log2(c.codebook_size) * c.frame_rate
    return max(1, math.floor(kbps * 1000 / per_codebook))
`;export{e as default};
