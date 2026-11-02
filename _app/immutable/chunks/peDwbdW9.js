const e=`# Constrained decoding with Qwen3-1.7B, as plain PyTorch and Python.
# A JSON schema becomes a regular expression (outlines-core's rules), the expression an automaton over bytes, and the
# automaton a mask over the vocabulary: at each step only the tokens whose bytes keep the text a prefix of some match
# may be chosen. Every other logit is set to −∞ before the argmax, so the probability the model gave them is removed
# and the rest renormalised. The end token is allowed only once the text is a full match. Each "# %% [id]" section is
# one card of the tab; outlines-core (a Rust library) does the same with a precomputed index.
import json
from dataclasses import dataclass

import torch


@dataclass
class Config:
    vocab_size: int = 151936  # rows of the LM head: 151,643 byte-level tokens, 26 control tokens, padding rows
    eos: int = 151645  # "<|im_end|>", allowed once the JSON is complete
    max_new_tokens: int = 160
    whitespace: str = "[ ]?"  # outlines-core's default: at most one space between JSON tokens


# %% [schema] JSON schema → regular expression
STRING_INNER = r'([^"\\\\\\x00-\\x1F\\x7F-\\x9F]|\\\\["\\\\/bfnrt])'  # any character but " \\ and control codes, or an escape
TYPES = {
    "string": f'"{STRING_INNER}*"',
    "integer": r"(-)?(0|[1-9][0-9]*)",
    "number": r"((-)?(0|[1-9][0-9]*))(\\.[0-9]+)?([eE][+-][0-9]+)?",
    "boolean": r"(true|false)",
    "null": "null",
}
META = set("\\\\.+*?()|[]{}^$#&-~")


def escape(text):
    return "".join("\\\\" + ch if ch in META else ch for ch in text)


def schema_regex(schema, ws=Config.whitespace):
    """A JSON schema → the regular expression of every JSON text it accepts, written as outlines-core writes it:
    an object's properties in the schema's order, all required, at most one space around : and ,."""
    if "enum" in schema:
        return "(" + "|".join(escape(json.dumps(v, ensure_ascii=False)) for v in schema["enum"]) + ")"
    kind = schema["type"]
    if kind == "object":
        assert set(schema.get("required", [])) == set(schema["properties"]), "only required properties"
        members = [f'"{escape(key)}"{ws}:{ws}{schema_regex(value, ws)}' for key, value in schema["properties"].items()]
        return r"\\{" + ws + f"{ws},{ws}".join(members) + ws + r"\\}"
    if kind == "array":
        item = f"({schema_regex(schema['items'], ws)})"
        return r"\\[" + ws + f"({item}(,{ws}{item}){{0,}})?" + ws + r"\\]"
    if kind == "string" and "maxLength" in schema:
        return f'"{STRING_INNER}{{{schema.get("minLength", 0)},{schema["maxLength"]}}}"'
    return TYPES[kind]


# %% [fsm] Regular expression → byte automaton
def parse(regex):
    """The expression → a tree: ("set", code-point ranges), ("cat", parts), ("alt", options), ("rep", node, lo, hi)."""
    pos = 0

    def peek():
        return regex[pos] if pos < len(regex) else ""

    def take():
        nonlocal pos
        pos += 1
        return regex[pos - 1]

    def escaped():
        ch = take()
        if ch == "x":
            return int(take() + take(), 16)
        return ord(ch)

    def char_class():
        negate = peek() == "^"
        if negate:
            take()
        ranges = []
        while peek() != "]":
            lo = escaped() if take() == "\\\\" else ord(regex[pos - 1])
            hi = lo
            if peek() == "-" and regex[pos + 1] != "]":
                take()
                hi = escaped() if take() == "\\\\" else ord(regex[pos - 1])
            ranges.append((lo, hi))
        take()
        if not negate:
            return ranges
        out, start = [], 0
        for lo, hi in sorted(ranges):
            if lo > start:
                out.append((start, lo - 1))
            start = max(start, hi + 1)
        return out + [(start, 0x10FFFF)] if start <= 0x10FFFF else out

    def atom():
        ch = take()
        if ch == "(":
            node = alternation()
            take()  # ")"
            return node
        if ch == "[":
            return ("set", char_class())
        code = escaped() if ch == "\\\\" else ord(ch)
        return ("set", [(code, code)])

    def repeat():
        node = atom()
        while peek() in ("?", "*", "+", "{"):
            ch = take()
            if ch == "{":
                body = ""
                while peek() != "}":
                    body += take()
                take()
                lo, _, hi = body.partition(",")
                node = ("rep", node, int(lo), int(hi) if hi else (None if "," in body else int(lo)))
            else:
                node = ("rep", node, 1 if ch == "+" else 0, 1 if ch == "?" else None)
        return node

    def concatenation():
        parts = []
        while peek() not in ("", "|", ")"):
            parts.append(repeat())
        return ("cat", parts)

    def alternation():
        options = [concatenation()]
        while peek() == "|":
            take()
            options.append(concatenation())
        return options[0] if len(options) == 1 else ("alt", options)

    return alternation()


def utf8_sequences(lo, hi):
    """Code points lo…hi → UTF-8 byte-range sequences that encode exactly them (surrogates skipped)."""
    out, stack = [], [(lo, hi)]
    while stack:
        s, e = stack.pop()
        if s > e:
            continue
        if s <= 0xDFFF and e >= 0xD800:  # surrogates have no UTF-8 encoding
            stack += [(s, 0xD7FF), (0xE000, e)]
            continue
        bound = next((b for b in (0x7F, 0x7FF, 0xFFFF) if s <= b < e), None)
        if bound is not None:  # one encoded length per sequence
            stack += [(s, bound), (bound + 1, e)]
            continue
        for i in range(1, len(chr(s).encode())):  # split until every byte varies independently
            m = (1 << (6 * i)) - 1
            if s & ~m != e & ~m:
                if s & m:
                    stack += [(s, s | m), ((s | m) + 1, e)]
                    break
                if e & m != m:
                    stack += [(s, (e & ~m) - 1), (e & ~m, e)]
                    break
        else:
            out.append(list(zip(chr(s).encode(), chr(e).encode())))
    return out


class Automaton:
    """A Thompson NFA over bytes, run as a DFA built on demand: a DFA state is the set of NFA states the text so far
    can be in. The empty set is dead: no continuation matches."""

    def __init__(self, regex):
        self.eps, self.edges = [], []  # per NFA state: ε-targets, and (lo, hi, target) byte ranges
        start, self.accept = self.build(parse(regex))
        self.sets, self.ids, self.moves = [], {}, {}
        self.start = self.state({start})

    def new(self):
        self.eps.append([])
        self.edges.append([])
        return len(self.eps) - 1

    def build(self, node):
        """A sub-automaton for one tree node → (its entry, its exit)."""
        kind = node[0]
        if kind == "set":
            a, b = self.new(), self.new()
            for lo, hi in node[1]:
                for seq in utf8_sequences(lo, hi):
                    at = a
                    for k, (x, y) in enumerate(seq):
                        to = b if k == len(seq) - 1 else self.new()
                        self.edges[at].append((x, y, to))
                        at = to
            return a, b
        if kind == "cat":
            a = b = self.new()
            for part in node[1]:
                s, e = self.build(part)
                self.eps[b].append(s)
                b = e
            return a, b
        if kind == "alt":
            a, b = self.new(), self.new()
            for option in node[1]:
                s, e = self.build(option)
                self.eps[a].append(s)
                self.eps[e].append(b)
            return a, b
        _, child, lo, hi = node  # "rep": lo copies, then hi − lo optional ones (or a loop)
        a = b = self.new()
        for _ in range(lo):
            s, e = self.build(child)
            self.eps[b].append(s)
            b = e
        if hi is None:
            s, e = self.build(child)
            self.eps[b].append(s)
            self.eps[e].append(s)
            end = self.new()
            self.eps[b].append(end)
            self.eps[e].append(end)
            return a, end
        end = self.new()
        for _ in range(hi - lo):
            self.eps[b].append(end)
            s, e = self.build(child)
            self.eps[b].append(s)
            b = e
        self.eps[b].append(end)
        return a, end

    def state(self, nfa_states):
        """The ε-closure of a set of NFA states, as a DFA state id."""
        todo, seen = list(nfa_states), set(nfa_states)
        while todo:
            for t in self.eps[todo.pop()]:
                if t not in seen:
                    seen.add(t)
                    todo.append(t)
        key = frozenset(seen)
        if key not in self.ids:
            self.ids[key] = len(self.sets)
            self.sets.append(key)
        return self.ids[key]

    def step(self, state, byte):
        if (state, byte) not in self.moves:
            targets = {t for s in self.sets[state] for lo, hi, t in self.edges[s] if lo <= byte <= hi}
            self.moves[state, byte] = self.state(targets)
        return self.moves[state, byte]

    def walk(self, state, data):
        """The state after reading \`data\`, or None once it is dead."""
        for byte in data:
            state = self.step(state, byte)
            if not self.sets[state]:
                return None
        return state

    def is_final(self, state):
        return self.accept in self.sets[state]


# %% [mask] The token mask
def allowed_tokens(automaton, state, vocab, c: Config = Config()):
    """vocab: {id: the token's bytes} for the tokens the grammar may use (control tokens left out). The ids whose
    bytes keep the text alive, plus the end token if the text so far is already a full match."""
    ids = [i for i, data in vocab.items() if automaton.walk(state, data) is not None]
    return sorted(ids + ([c.eos] if automaton.is_final(state) else []))


def mask_logits(logits, allowed):
    """logits (V,) → the masked logits: −∞ outside the allowed ids."""
    masked = torch.full_like(logits, float("-inf"))
    masked[allowed] = logits[allowed]
    return masked


def removed_mass(logits, allowed):
    """The probability the unmasked softmax gave to the tokens the mask removes."""
    p = logits.double().softmax(-1)
    return 1 - float(p[allowed].sum())


# %% [model] Constrained decoding
@torch.no_grad()
def constrained_greedy(model, prompt_ids, automaton, vocab, c: Config = Config()):
    """Greedy decoding with the mask: at each step the most likely allowed token. Stops at the end token, which the
    mask allows only in a final state. Returns the tokens and, per step, the probability mass removed."""
    ids, out, removed = list(prompt_ids), [], []
    state = automaton.start
    for _ in range(c.max_new_tokens):
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
        allowed = allowed_tokens(automaton, state, vocab, c)
        token = int(mask_logits(logits, allowed).argmax())
        removed.append(removed_mass(logits, allowed))
        ids.append(token)
        out.append(token)
        if token == c.eos:
            break
        state = automaton.walk(state, vocab[token])
    return out, removed


# schema = {"type": "object", "properties": {"name": {"type": "string"}, …}, "required": [...]}
# automaton = Automaton(schema_regex(schema)); vocab = {id: bytes of the token} for ids < 151,643
# tokens, removed = constrained_greedy(model, prompt_ids, automaton, vocab)
`;export{e as default};
