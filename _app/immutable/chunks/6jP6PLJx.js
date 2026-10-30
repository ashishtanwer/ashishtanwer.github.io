const e=`# o200k_base, the tokenizer of GPT-4o (OpenAI 2024), from tiktoken's o200k_base.tiktoken, as plain Python.
# Byte-level BPE like GPT-2's, with three differences: the vocabulary is four times larger (199,998 byte strings); the
# pre-tokenizer regex cuts words where lower case turns back to upper, keeps contractions on their word and splits
# digits into groups of up to three; and the merges are not a separate list: a byte string's rank is its token id, and
# the adjacent pair whose concatenation has the lowest rank merges first, one at a time.
# Each "# %% [id]" section is one step of the o200k tab. The only package used is regex, for \\p{…} classes.
import base64
from dataclasses import dataclass
from pathlib import Path

import regex


@dataclass
class Config:
    vocab_size: int = 199998  # ranked byte strings; ids 199999 and 200018 are special tokens
    endoftext_id: int = 199999  # <|endoftext|>, never produced from text


def load(path):
    """o200k_base.tiktoken: one "base64(bytes) rank" line per token, ranks 0 … 199997."""
    ranks = {}
    for line in Path(path).read_bytes().splitlines():
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return ranks


# %% [pretokenize] Pre-tokenizer: seven alternatives, tried left to right
ALTERNATIVES = [
    # a word ending in lower-case letters (Token, naïve, 東京は: Lo counts as both cases), with at most one leading
    # space or symbol and a contraction kept on it
    r"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    # an upper-case word (GPT in GPTs, HELLO'S)
    r"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\\p{N}{1,3}",  # digits, at most three at a time and never with a space: 12345 → 123 45
    r" ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",  # punctuation and symbols, with one leading space
    r"\\s*[\\r\\n]+",  # line breaks
    r"\\s+(?!\\S)",  # spaces before a word, leaving the last one to the word
    r"\\s+",
]
PATTERN = regex.compile("|".join(ALTERNATIVES))


def pretokenize(text):
    return [match.group() for match in PATTERN.finditer(text)]


def branch(text, start):
    """Which alternative cut the piece at \`start\`: the regex tries them in order and keeps the first that matches."""
    return next(i for i, alternative in enumerate(ALTERNATIVES) if regex.compile(alternative).match(text, start))


# %% [bpe] BPE: merge the pair whose bytes have the lowest rank
def bpe(piece, ranks, steps=None):
    """piece: the UTF-8 bytes of one pre-token. Start from single bytes; each step merges the one adjacent pair whose
    concatenation has the lowest rank (the leftmost on a tie), until no concatenation is in the vocabulary. This is
    tiktoken's own _educational.bpe_encode."""
    parts = [bytes([b]) for b in piece]
    while len(parts) > 1:
        rank, i = min((ranks.get(a + b, float("inf")), i) for i, (a, b) in enumerate(zip(parts, parts[1:])))
        if rank == float("inf"):
            break
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2 :]
        if steps is not None:
            steps.append((i, rank, list(parts)))
    return [ranks[part] for part in parts]


# %% [encode] Encode: pieces → bytes → ids
def encode(text, ranks):
    return [i for piece in pretokenize(text) for i in bpe(piece.encode("utf-8"), ranks)]


# %% [decode] Decode: ids → bytes → the exact text
def decode(ids, ranks):
    tokens = {rank: token for token, rank in ranks.items()}
    return b"".join(tokens[i] for i in ids).decode("utf-8", errors="replace")
`;export{e as default};
