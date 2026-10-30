const n=`# GGUF and llama.cpp's k-quants, the format in the Efficiency Explainer's GGUF tab, written as plain PyTorch.
# Qwen's own qwen2.5-0.5b-instruct-q4_k_m.gguf (Qwen/Qwen2.5-0.5B-Instruct-GGUF) is a "Q4_K_M" file: a mix of block
# types chosen per tensor. Q4_K packs 256 weights into 144 bytes: 8 sub-blocks of 32, each with a 6-bit scale and a
# 6-bit min, times one fp16 d and dmin per super-block. Each "# %% [id]" section is one block of the page.
import struct
from dataclasses import dataclass

import torch

QK_K = 256  # weights per k-quant super-block


@dataclass
class Config:  # bytes and weights per block of the four types in the Q4_K_M file
    q4_k: tuple = (144, 256)  # fp16 d, fp16 dmin, 12 bytes of 6-bit scales and mins, 128 bytes of 4-bit codes
    q6_k: tuple = (210, 256)  # 128 bytes of low 4 bits, 64 of high 2 bits, 16 int8 scales, fp16 d
    q5_0: tuple = (22, 32)  # fp16 d, 4 bytes of fifth bits, 16 bytes of low 4 bits
    q8_0: tuple = (34, 32)  # fp16 d, 32 int8 codes


TYPES = {0: "F32", 1: "F16", 6: "Q5_0", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K"}  # ggml type ids in this file


# %% [file] The GGUF container
def read_gguf(path):
    """The header of a GGUF (v3) file: metadata {key: value} and tensors {name: (shape, type, absolute offset)}.
    Shapes are ggml's, fastest dimension first: a weight (out, in) is listed as (in, out)."""
    with open(path, "rb") as f:
        read = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        string = lambda: f.read(read("Q")).decode("utf-8")
        scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:  # an array: element type, count, elements
                inner, count = read("I"), read("Q")
                return [value(inner) for _ in range(count)]
            return read(scalars[kind])

        assert f.read(4) == b"GGUF" and read("I") == 3
        n_tensors, n_metadata = read("Q"), read("Q")
        metadata = {}
        for _ in range(n_metadata):
            key = string()
            metadata[key] = value(read("I"))
        infos = []
        for _ in range(n_tensors):
            name = string()
            shape = tuple(read("Q") for _ in range(read("I")))
            infos.append((name, shape, read("I"), read("Q")))
        alignment = metadata.get("general.alignment", 32)
        start = -(-f.tell() // alignment) * alignment  # tensor data begins at the next aligned byte
    return metadata, {name: (shape, kind, start + offset) for name, shape, kind, offset in infos}


def fp16(two_bytes):  # (..., 2) uint8 → (...) float32
    return two_bytes.contiguous().view(torch.float16).squeeze(-1).float()


# %% [q8_0] Q8_0: 32 weights, one scale
def dequantize_q8_0(blocks):
    """blocks (n, 34) uint8 → (n, 32): w = d · q with q an int8."""
    d = fp16(blocks[:, :2])
    q = blocks[:, 2:].contiguous().view(torch.int8).float()
    return d[:, None] * q


# %% [q5_0] Q5_0: 32 weights of 5 bits
def dequantize_q5_0(blocks):
    """blocks (n, 22) uint8 → (n, 32): w = d · (q − 16). Weight j's low 4 bits are the low nibble of byte j
    (j < 16) or the high nibble of byte j − 16; its fifth bit is bit j of the 32-bit qh."""
    d = fp16(blocks[:, :2])
    qh = blocks[:, 2:6].to(torch.int64)
    qh = qh[:, 0] | qh[:, 1] << 8 | qh[:, 2] << 16 | qh[:, 3] << 24
    qs = blocks[:, 6:].to(torch.int64)
    low = torch.cat([qs & 0x0F, qs >> 4], dim=1)  # (n, 32)
    high = (qh[:, None] >> torch.arange(32)) & 1
    return (low | high << 4).sub(16).float() * d[:, None]


# %% [q4_k] Q4_K: super-blocks of 256 with 6-bit scales and mins
def scale_min(scales):
    """12 bytes → eight 6-bit scales and eight 6-bit mins. Sub-blocks 0–3 use the low 6 bits of bytes 0–3 (scales)
    and 4–7 (mins); sub-blocks 4–7 take 4 low bits from bytes 8–11 and their top 2 bits from the spare top bits of
    bytes 0–7."""
    s = scales.to(torch.int64)
    d, m, md = s[:, 0:4], s[:, 4:8], s[:, 8:12]
    sc = torch.cat([d & 63, (md & 0x0F) | ((d >> 6) << 4)], dim=1)
    mn = torch.cat([m & 63, (md >> 4) | ((m >> 6) << 4)], dim=1)
    return sc, mn  # (n, 8) each, integers 0 … 63


def dequantize_q4_k(blocks):
    """blocks (n, 144) uint8 → (n, 256): sub-block j (32 weights) is w = d · sc[j] · q − dmin · m[j] with q in 0 … 15.
    Each 32 bytes of codes hold two sub-blocks: the low nibbles are sub-block 2k, the high nibbles 2k + 1."""
    d, dmin = fp16(blocks[:, 0:2]), fp16(blocks[:, 2:4])
    sc, mn = scale_min(blocks[:, 4:16])
    qs = blocks[:, 16:].reshape(-1, 4, 1, 32).to(torch.int64)
    q = ((qs >> torch.tensor([0, 4]).view(1, 1, 2, 1)) & 0x0F).reshape(-1, 8, 32).float()
    step = (d[:, None] * sc.float())[:, :, None]  # d · sc[j]
    offset = (dmin[:, None] * mn.float())[:, :, None]  # dmin · m[j]
    return (step * q - offset).reshape(-1, QK_K)


# %% [q6_k] Q6_K: 6-bit codes, 16 sub-blocks of 16
def dequantize_q6_k(blocks):
    """blocks (n, 210) uint8 → (n, 256): w = d · scale[j] · (q − 32) with q's low 4 bits in ql, its top 2 bits in qh,
    and one int8 scale per 16 weights."""
    ql, qh = blocks[:, :128].to(torch.int64), blocks[:, 128:192].to(torch.int64)
    scales = blocks[:, 192:208].contiguous().view(torch.int8).float()
    d = fp16(blocks[:, 208:210])
    ql = ql.reshape(-1, 2, 1, 64) >> torch.tensor([0, 4]).view(1, 1, 2, 1)  # (n, half, nibble, 64)
    ql = (ql & 0x0F).reshape(-1, 2, 2, 2, 32)  # (n, half, nibble, first/second 32 bytes, 32)
    qh = qh.reshape(-1, 2, 1, 32) >> torch.tensor([0, 2, 4, 6]).view(1, 1, 4, 1)
    qh = (qh & 3).reshape(-1, 2, 2, 2, 32)  # bit pair k = 2 · nibble + first/second 32 bytes
    q = (ql | qh << 4).reshape(-1, QK_K).float() - 32  # 32 weights each from k = 0 … 3, per half
    step = (d[:, None] * scales).reshape(-1, 16, 1)
    return (step * q.reshape(-1, 16, 16)).reshape(-1, QK_K)


# %% [tensor] A tensor's bytes → its weights
DEQUANTIZE = {"Q8_0": dequantize_q8_0, "Q5_0": dequantize_q5_0, "Q4_K": dequantize_q4_k, "Q6_K": dequantize_q6_k}


def read_tensor(path, shape, kind, offset):
    """The weights of one tensor as a torch matrix in PyTorch's order (out, in): ggml rows are contiguous runs of
    blocks along the first (input) dimension."""
    count = 1
    for n in shape:
        count *= n
    name = TYPES[kind]
    if name in ("F32", "F16"):
        size, dtype = (4, torch.float32) if name == "F32" else (2, torch.float16)
        with open(path, "rb") as f:
            f.seek(offset)
            data = torch.frombuffer(bytearray(f.read(count * size)), dtype=dtype).float()
    else:
        size, per = getattr(Config(), name.lower())
        with open(path, "rb") as f:
            f.seek(offset)
            raw = torch.frombuffer(bytearray(f.read(count // per * size)), dtype=torch.uint8)
        data = DEQUANTIZE[name](raw.reshape(-1, size))
    return data.reshape(tuple(reversed(shape)))


# %% [model] The whole file as a Qwen2 state dict
NAMES = {
    "token_embd": "model.embed_tokens", "output_norm": "model.norm", "output": "lm_head",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}


def hf_name(name):
    """llama.cpp's tensor names → transformers' Qwen2 names, e.g. blk.12.ffn_down.weight → model.layers.12.mlp.down_proj.weight."""
    parts = name.split(".")
    if parts[0] == "blk":
        return f"model.layers.{parts[1]}.{NAMES[parts[2]]}.{parts[3]}"
    return f"{NAMES[parts[0]]}.{parts[1]}"


def state_dict(path):
    """Every tensor of the file dequantized to float32 under its transformers name."""
    _, tensors = read_gguf(path)
    return {hf_name(name): read_tensor(path, *info) for name, info in tensors.items()}
`;export{n as default};
