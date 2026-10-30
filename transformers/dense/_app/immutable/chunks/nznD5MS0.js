const e=`# FAST (Pertsch et al. 2025), physical-intelligence/fast, as plain Python and NumPy.
# An action chunk (here 50 steps × 7 numbers) is normalised per joint so the dataset's 1st and 99th percentiles land
# on −1 and +1, turned into frequencies with a DCT along time, scaled ×10 and rounded: most high frequencies become 0.
# The integers, low frequencies first, are written as characters, and a byte-level BPE with 2048 tokens merges
# frequent runs (long runs of zeros above all) into single tokens. Decoding runs the steps backwards; only the
# rounding loses anything. Each "# %% [id]" section is one step of the FAST tab.
import math
import unicodedata
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    time: int = 50  # actions per chunk
    dims: int = 7  # LIBERO: Δx, Δy, Δz, three rotation Δ, gripper
    scale: float = 10.0  # DCT coefficients ×10 before rounding
    min_token: int = -354  # the smallest rounded coefficient seen when the vocabulary was fit
    vocab: int = 2048


# %% [normalize] Quantile normalisation
def normalize(actions, q01, q99):
    """openpi's quantile normalisation: each joint's 1st and 99th percentile over the dataset map to −1 and +1.
    Values beyond them are not clamped; FAST encodes them as they are."""
    return (actions - q01) / (q99 - q01 + 1e-6) * 2 - 1


def denormalize(x, q01, q99):
    return (x + 1) / 2 * (q99 - q01 + 1e-6) + q01


# %% [dct] DCT along time, per joint
def dct_matrix(n):
    """Orthonormal DCT-II: row k samples cos(π k (2t + 1) / 2n). The rows are orthonormal, so the inverse DCT is
    the transpose."""
    k, t = np.arange(n)[:, None], np.arange(n)[None]
    m = np.cos(math.pi * k * (2 * t + 1) / (2 * n)) * math.sqrt(2 / n)
    m[0] /= math.sqrt(2)
    return m


def dct(x):
    """x: (time, dims) → coefficients (frequency, dims). Frequency 0 is the joint's mean × √time."""
    return dct_matrix(x.shape[0]) @ x


# %% [quantize] ×10, round, flatten, shift
def quantize(coefficients, c: Config):
    """Scaled and rounded (half to even), then read row by row: frequency 0 of every joint, then frequency 1, …
    Shifted by −min_token so every integer is a non-negative character code (0 → chr(354), 'Ţ')."""
    rounded = np.around(coefficients * c.scale)
    codes = np.maximum(rounded.flatten() - c.min_token, 0).astype(int)
    return rounded, "".join(map(chr, codes))


# %% [bytes] Characters → bytes → BPE's alphabet
def bytes_to_unicode():
    """GPT-2's table: each of the 256 byte values as a printable character (unprintable bytes move up past 255)."""
    keep = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    table, extra = {b: chr(b) for b in keep}, 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + extra)
            extra += 1
    return table


BYTE = bytes_to_unicode()
# Unicode's White_Space characters, which the pre-tokenizer's \\s matches.
SPACE = {*range(9, 14), 32, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000}


def kind(ch):
    if ord(ch) in SPACE:
        return "space"
    return {"L": "letter", "N": "number"}.get(unicodedata.category(ch)[0], "other")


def pre_tokenize(text):
    """The byte-level pre-tokenizer's split, GPT-2's pattern: 's 't … contractions, then ' ?letters', ' ?numbers',
    ' ?other', and runs of spaces (a run before a word leaves its last space to that word). Near code 354 (an
    integer of 0) every character is a Latin letter, so a chunk is almost always one word."""
    words, i = [], 0
    while i < len(text):
        start = i
        contraction = next((s for s in ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d") if text.startswith(s, i)), None)
        if contraction:
            i += len(contraction)
        else:
            j = i + 1 if text[i] == " " and i + 1 < len(text) and kind(text[i + 1]) != "space" else i
            if kind(text[j]) != "space":
                group = kind(text[j])
                i = j + 1
                while i < len(text) and kind(text[i]) == group:
                    i += 1
            else:
                while i < len(text) and kind(text[i]) == "space":
                    i += 1
                if i < len(text) and i - start > 1:
                    i -= 1  # \\s+(?!\\S): the last space goes with the next word
        words.append(text[start:i])
    return words


def to_alphabet(word):
    """A word's UTF-8 bytes in BPE's alphabet: every character from 128 to 2047 is two bytes."""
    return "".join(BYTE[b] for b in word.encode("utf-8"))


# %% [bpe] Merges, lowest rank first
def load_bpe(tokenizer):
    """vocab (piece → id) and each merge's rank from the published tokenizer.json (a dict)."""
    model = tokenizer["model"]
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in model["merges"]]
    return model["vocab"], {pair: rank for rank, pair in enumerate(merges)}


def bpe(word, ranks):
    """Start from single bytes; repeatedly take the adjacent pair whose merge has the lowest rank and merge every
    occurrence of it, left to right, until no adjacent pair has a merge. Returns the pieces and, per merge, its
    rank, its pair and the piece indices it merged at."""
    pieces, steps = list(word), []
    while len(pieces) > 1:
        pairs = {pair for pair in zip(pieces, pieces[1:]) if pair in ranks}
        if not pairs:
            break
        best = min(pairs, key=ranks.__getitem__)
        merged, at, i = [], [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == best:
                merged.append(pieces[i] + pieces[i + 1])
                at.append(i)
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        steps.append((ranks[best], best, at))
        pieces = merged
    return pieces, steps


def encode_text(text, vocab, ranks):
    return [vocab[piece] for word in pre_tokenize(text) for piece in bpe(to_alphabet(word), ranks)[0]]


# %% [decode] Tokens → actions
def decode(ids, vocab, c: Config, q01=None, q99=None):
    """Pieces → bytes → UTF-8 text → integers + min_token → ÷10 → inverse DCT (the transpose) → actions."""
    piece = {i: p for p, i in vocab.items()}
    byte = {ch: b for b, ch in BYTE.items()}
    text = bytes(byte[ch] for i in ids for ch in piece[i]).decode("utf-8", errors="replace")
    rounded = (np.array([ord(ch) for ch in text]) + c.min_token).reshape(-1, c.dims)
    x = dct_matrix(len(rounded)).T @ (rounded / c.scale)
    return x if q01 is None else denormalize(x, q01, q99)


# %% [model] FAST
class FAST:
    """The whole tokenizer: actions in environment units ↔ token ids."""

    def __init__(self, tokenizer, c: Config = Config()):
        self.c = c
        self.vocab, self.ranks = load_bpe(tokenizer)

    def encode(self, actions, q01, q99):
        _, text = quantize(dct(normalize(actions, q01, q99)), self.c)
        return encode_text(text, self.vocab, self.ranks)

    def decode(self, ids, q01, q99):
        return decode(ids, self.vocab, self.c, q01, q99)
`;export{e as default};
