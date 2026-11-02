const e=`# SLERP merging (spherical linear interpolation, Shoemake, 1985; mergekit's \`slerp\` method) on the Post-training
# Explainer's SLERP tab, as plain PyTorch: Qwen2.5-0.5B-Instruct and Qwen2.5-Coder-0.5B-Instruct, two fine-tunes of
# Qwen2.5-0.5B, interpolated tensor by tensor. Each tensor is treated as one long vector: the angle Ω between the two
# models' vectors sets how far along the arc between them a share t lands, while a linear average walks the chord.
import math
from dataclasses import dataclass

import torch


@dataclass
class Config:
    models: tuple = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    t: float = 0.5  # 0 is the first model, 1 the second
    dot_threshold: float = 0.9995  # cos Ω above this counts as parallel: use lerp (mergekit's value)


# %% [angle] The angle between two models
def angle(v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> tuple[float, float]:
    """cos Ω = ⟨v0, v1⟩ / (‖v0‖ ‖v1‖) of two tensors flattened to vectors, and Ω = arccos(cos Ω) in radians."""
    a, b = v0.flatten().double(), v1.flatten().double()
    a = a / a.norm() if a.norm() > eps else a
    b = b / b.norm() if b.norm() > eps else b
    dot = float(a @ b)
    return dot, math.acos(max(-1.0, min(1.0, dot)))


# %% [lerp] Linear interpolation
def lerp(t: float, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """(1 − t)·v0 + t·v1: the straight chord between the two tensors, what a plain weighted average gives."""
    return (1 - t) * v0.float() + t * v1.float()


# %% [slerp] Spherical interpolation
def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """sin((1 − t)Ω)/sin Ω · v0 + sin(tΩ)/sin Ω · v1, with Ω the angle between the tensors' directions. The weights
    move a unit vector along the great circle at a constant angular speed; as in mergekit they are applied to the
    tensors themselves, so the result's length follows the two lengths. Nearly parallel tensors use lerp."""
    dot, omega = angle(v0, v1)
    if abs(dot) > dot_threshold:
        return lerp(t, v0, v1)
    s0 = math.sin((1 - t) * omega) / math.sin(omega)
    s1 = math.sin(t * omega) / math.sin(omega)
    return s0 * v0.float() + s1 * v1.float()


# %% [merge] Merging two models
@torch.no_grad()
def slerp_merge(first: dict, second: dict, t: float, dot_threshold: float = 0.9995) -> dict:
    """Every tensor interpolated on its own, with one t for the whole model."""
    return {name: slerp(t, v0, second[name], dot_threshold) for name, v0 in first.items()}


def lerp_merge(first: dict, second: dict, t: float) -> dict:
    """The linear baseline at the same t (t = 0.5 is the plain average of the two models)."""
    return {name: lerp(t, v0, second[name]) for name, v0 in first.items()}
`;export{e as default};
