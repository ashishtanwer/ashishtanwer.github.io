const n=`# ByT5 (google/byt5-small): no vocabulary to learn. Every UTF-8 byte is one token, shifted by 3 so that
# ids 0, 1 and 2 stay free for <pad>, </s> and <unk>; the model then looks each id up in a 384-row table.
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Config:
    pad_id: int = 0
    eos_id: int = 1
    unk_id: int = 2
    offset: int = 3  # byte b has id b + 3
    extra_ids: int = 125  # <extra_id_0> … <extra_id_124>, ids 259 … 383 in transformers' tokenizer
    vocab_size: int = 384  # 3 special + 256 bytes + 125 extra ids
    d_model: int = 1472


# %% [bytes] UTF-8 bytes
def utf8_bytes(text: str) -> list[int]:
    """UTF-8 by hand: 1 byte below U+0080, else a lead byte (110…, 1110…, 11110…) carrying the high bits and
    6 more bits in each continuation byte (10…). Python's text.encode('utf-8') gives the same."""
    out = []
    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            out.append(cp)
        elif cp < 0x800:
            out += [0xC0 | cp >> 6, 0x80 | cp & 0x3F]
        elif cp < 0x10000:
            out += [0xE0 | cp >> 12, 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F]
        else:
            out += [0xF0 | cp >> 18, 0x80 | cp >> 12 & 0x3F, 0x80 | cp >> 6 & 0x3F, 0x80 | cp & 0x3F]
    return out


# %% [ids] Byte ids
def encode(text: str, config: Config = Config()) -> list[int]:
    """One id per byte, then </s>. Nothing is ever unknown: every byte has an id."""
    return [b + config.offset for b in utf8_bytes(text)] + [config.eos_id]


# %% [decode] Decoding
def decode(ids: list[int], config: Config = Config()) -> str:
    """Drop the special and extra ids, subtract 3, and read the bytes as UTF-8. A character cut off midway
    (its lead byte without all its continuation bytes) is dropped, as errors='ignore' does."""
    data = bytes(i - config.offset for i in ids if config.offset <= i < config.offset + 256)
    return data.decode('utf-8', errors='ignore')


# %% [embedding] Embedding lookup
class ByteEmbedding(nn.Module):
    """ByT5's \`shared\` table, used by the encoder and the decoder alike. T5 adds no position vector (its
    attention has a relative position bias) and does not scale the rows."""

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # (batch, length) → (batch, length, d_model)
        return self.shared(ids)


# %% [model] Text to the encoder's input
def byt5_input(text: str, embedding: ByteEmbedding, config: Config = Config()) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([encode(text, config)])  # (1, bytes + 1)
    return ids, embedding(ids)  # the vectors the first encoder layer reads
`;export{n as default};
