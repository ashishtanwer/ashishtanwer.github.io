const e=`# BERT's WordPiece tokenizer (Devlin et al. 2019), google-bert/bert-base-uncased vocab.txt, as plain Python.
# The text is cleaned, Chinese characters are spaced out, accents are stripped and everything is lower-cased; then it is
# split on whitespace and punctuation, and each word is cut greedily into the longest vocabulary pieces from the left,
# pieces after the first marked with ##. A word with any position no piece covers becomes [UNK] as a whole. [CLS] and
# [SEP] wrap the result. Decoding gives back the normalized text, not the input.
# Each "# %% [id]" section is one step of the BERT tab.
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    vocab_size: int = 30522
    max_chars_per_word: int = 100  # longer words become [UNK] without trying
    prefix: str = "##"  # marks a piece that continues a word
    hidden_size: int = 768  # width of the model's token-embedding rows


def load(folder):
    """vocab.txt holds one piece per line; the line number is its id ([PAD] 0, [UNK] 100, [CLS] 101, [SEP] 102)."""
    lines = (Path(folder) / "vocab.txt").read_text(encoding="utf-8").split("\\n")
    return {piece: i for i, piece in enumerate(lines) if piece}


# %% [normalize] Normalize: clean, space out Chinese characters, strip accents, lower-case
def is_chinese(c):
    """The CJK Unified Ideograph blocks only; kana and hangul are ordinary letters."""
    cp = ord(c)
    return any(
        start <= cp <= end
        for start, end in [
            (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F),
        ]
    )


def normalize(text):
    out = []
    for c in text:
        if c in "\\t\\n\\r" or c.isspace():
            out.append(" ")
        elif ord(c) == 0 or ord(c) == 0xFFFD or unicodedata.category(c).startswith("C"):
            continue  # control and format characters are dropped
        elif is_chinese(c):
            out.append(f" {c} ")  # each ideograph becomes its own word
        else:
            out.append(c)
    text = unicodedata.normalize("NFD", "".join(out))  # é → e + U+0301 ...
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")  # ... and the accent mark is dropped
    return text.lower()


# %% [split] Basic split: whitespace, then each punctuation mark on its own
def is_punctuation(c):
    cp = ord(c)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True  # all ASCII non-letters, non-digits: $ + < = > ^ \` | ~ count too
    return unicodedata.category(c).startswith("P")


def split(text):
    words = []
    for chunk in text.split():
        current = ""
        for c in chunk:
            if is_punctuation(c):
                words += [current, c] if current else [c]
                current = ""
            else:
                current += c
        if current:
            words.append(current)
    return words


# %% [wordpiece] WordPiece: longest vocabulary piece first, ## for the rest of a word
def wordpiece(word, vocab, config=Config(), tries=None):
    """Greedy, not optimal: from the left, take the longest piece in the vocabulary, then continue after it."""
    if len(word) > config.max_chars_per_word:
        return ["[UNK]"]
    pieces, start = [], 0
    while start < len(word):
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else config.prefix + word[start:end]
            if tries is not None:
                tries.append((start, end, piece in vocab))
            if piece in vocab:
                pieces.append(piece)
                start = end
                break
        else:
            return ["[UNK]"]  # nothing in the vocabulary starts here: the whole word is unknown
    return pieces


# %% [encode] Encode: [CLS] pieces [SEP] → ids
def encode(text, vocab):
    pieces = [piece for word in split(normalize(text)) for piece in wordpiece(word, vocab)]
    return [vocab[piece] for piece in ["[CLS]", *pieces, "[SEP]"]]


# %% [decode] Decode: glue ## pieces on, put a space before the others
# The released decoder cleans each piece on its own, so only " ." " ?" " !" " ," can ever match: "don't" comes back
# as "don ' t".
CLEANUP = [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
           (" do not", " don't"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]
SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")  # skipped, as decode(..., skip_special_tokens=True) does


def decode(ids, vocab):
    pieces = {i: piece for piece, i in vocab.items()}
    out = []
    for n, piece in enumerate(pieces[i] for i in ids if pieces[i] not in SPECIAL):
        if n > 0:
            piece = piece[len(Config.prefix):] if piece.startswith(Config.prefix) else " " + piece
        for before, after in CLEANUP:
            piece = piece.replace(before, after)
        out.append(piece)
    return "".join(out)
`;export{e as default};
