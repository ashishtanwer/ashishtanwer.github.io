const e=`# GPT-2's tokenizer (Radford et al. 2019), openai-community/gpt2 vocab.json + merges.txt, as plain Python.
# Byte-level BPE: the text's UTF-8 bytes are mapped one to one onto 256 printable characters, a regex cuts the text
# into word-like pieces, and inside each piece the lowest-ranked adjacent pair from merges.txt is merged again and
# again until no pair is left. Every vocabulary entry is a string of those characters, so any input encodes and
# decodes back exactly, and there is never an unknown token.
# Each "# %% [id]" section is one step of the GPT-2 tab. The only package used is regex, for \\p{L} and \\p{N}.
import json
from dataclasses import dataclass
from pathlib import Path

import regex


@dataclass
class Config:
    vocab_size: int = 50257  # 256 byte characters + 50,000 merges + <|endoftext|>
    merges: int = 50000
    eos_id: int = 50256  # <|endoftext|>; GPT-2's tokenizer adds no special token around a text
    n_embd: int = 768  # width of the model's token-embedding rows (wte is 50257 × 768)


def load(folder):
    """vocab.json maps each token string to its id; merges.txt lists the merges in rank order after a header line."""
    folder = Path(folder)
    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    lines = (folder / "merges.txt").read_text(encoding="utf-8").split("\\n")[1:]
    ranks = {tuple(line.split()): rank for rank, line in enumerate(line for line in lines if line)}
    return vocab, ranks


# %% [bytes] UTF-8 bytes → printable characters
def bytes_to_unicode():
    """Byte b → one printable character. The 188 printable bytes keep their own character; the other 68 (controls,
    space, 0x7F–0xA0, 0xAD) are moved to U+0100 and up, so a space becomes Ġ (U+0120) and a newline Ċ (U+010A)."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars = {b: chr(b) for b in printable}
    shift = 0
    for b in range(256):
        if b not in chars:
            chars[b] = chr(256 + shift)
            shift += 1
    return chars


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def to_chars(text):
    return "".join(BYTE_TO_CHAR[b] for b in text.encode("utf-8"))


# %% [pretokenize] Pre-tokenizer: the regex cuts the text into pieces
# Contractions, then letters, digits or other symbols each with at most one leading space, then runs of spaces. A run of
# spaces before a word gives its last space to the word (\\s+(?!\\S) stops one short).
PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+""")


def pretokenize(text):
    return [match.group() for match in PATTERN.finditer(text)]


# %% [bpe] BPE: merge the lowest-ranked pair until none is left
def bpe(piece, ranks, steps=None):
    """piece: one pre-token already in byte characters. Each round finds the adjacent pair with the lowest rank in
    merges.txt and merges every occurrence of it, left to right, as OpenAI's encoder.py does."""
    symbols = list(piece)
    while len(symbols) > 1:
        pairs = {(a, b) for a, b in zip(symbols, symbols[1:])}
        best = min(pairs, key=lambda pair: ranks.get(pair, float("inf")))
        if best not in ranks:
            break  # no adjacent pair is a merge any more
        merged, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        if steps is not None:
            steps.append((best, ranks[best], merged))
        symbols = merged
    return symbols


# %% [encode] Encode: pieces → token strings → ids
def encode(text, vocab, ranks):
    ids = []
    for piece in pretokenize(text):
        ids += [vocab[token] for token in bpe(to_chars(piece), ranks)]
    return ids


# %% [decode] Decode: ids → token strings → bytes → the exact text
def decode(ids, vocab):
    tokens = {i: token for token, i in vocab.items()}
    data = bytes(CHAR_TO_BYTE[c] for i in ids for c in tokens[i])
    return data.decode("utf-8", errors="replace")  # a cut-off multi-byte character shows as U+FFFD
`;export{e as default};
