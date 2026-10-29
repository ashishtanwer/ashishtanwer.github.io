const e=`# Latent consistency model (LCM-LoRA, latent-consistency/lcm-lora-sdv1-5) on Stable Diffusion 1.5, the model of the LCM
# tab, as plain PyTorch. A rank-64 LoRA merged into SD 1.5's U-Net turns it into a consistency model: from any noisy
# latent it jumps straight to a clean one, so 2–4 steps replace the 25–50 of SD 1.5's samplers.
# The U-Net, its blocks, the VAE and guidance are the SD 1.5 tab's (pytorch/diffusion_sd15.py).
# Each "# %% [id]" section is one step of the tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_sd15 import Config as SD15Config
from diffusion_sd15 import TimestepEmbedding, guide, noise_schedule, sinusoid


@dataclass
class Config(SD15Config):
    rank: int = 64  # every LoRA pair in the file is rank 64 …
    alpha: float = 8.0  # … with α = 8, so the update is scaled by α / r = 1/8
    original_steps: int = 50  # the teacher's DDIM grid the LoRA was distilled on: 999, 979, …, 19
    timestep_scaling: float = 10.0
    sigma_data: float = 0.5
    guidance_dim: int = 256  # full LCMs only: the width of the guidance-scale embedding


# %% [lora] Low-rank update
class LoRA(nn.Module):
    """A rank-r update beside a frozen layer: \`down\` maps the input to r numbers, \`up\` maps them back, scaled by
    α / r. For a k × k convolution, \`down\` is a k × k convolution to r channels (with the layer's stride and
    padding) and \`up\` a 1 × 1 convolution. Its weight is ΔW = (α / r) · up · down, rank r at most."""

    def __init__(self, layer: nn.Module, rank: int, alpha: float):
        super().__init__()
        if isinstance(layer, nn.Linear):
            self.down = nn.Linear(layer.in_features, rank, bias=False)
            self.up = nn.Linear(rank, layer.out_features, bias=False)
        else:
            self.down = nn.Conv2d(layer.in_channels, rank, layer.kernel_size, layer.stride, layer.padding, bias=False)
            self.up = nn.Conv2d(rank, layer.out_channels, 1, bias=False)
        self.scale = alpha / rank

    def forward(self, x):
        return self.scale * self.up(self.down(x))

    def delta(self):
        """ΔW in the layer's weight shape: (out, r) @ (r, in · k · k)."""
        up, down = self.up.weight.flatten(1), self.down.weight
        return (self.scale * up @ down.flatten(1)).view(up.shape[0], *down.shape[1:])


class WithLoRA(nn.Module):
    """The layer and its LoRA side by side, unmerged: y = W·x + b + s · ΔW·x. Costs a little per call."""

    def __init__(self, layer: nn.Module, lora: LoRA, scale: float = 1.0):
        super().__init__()
        self.layer, self.lora, self.scale = layer, lora, scale

    def forward(self, x):
        return self.layer(x) + self.scale * self.lora(x)


# %% [lora-merge] Merging the update into the weights
@torch.no_grad()
def merge_lora(model: nn.Module, loras: dict, scale: float = 1.0):
    """W ← W + s · ΔW for every layer the LoRA covers, named as in \`model\`. In LCM-LoRA that is 278 layers:
    attention q, k, v and out, the feed-forward, proj_in and proj_out, every ResNet convolution, shortcut and
    timestep projection, and the down- and upsamplers. Merged, the U-Net costs exactly what SD 1.5's does."""
    for name, lora in loras.items():
        model.get_submodule(name).weight += scale * lora.delta()


# %% [guidance-embedding] Guidance as an input (full LCMs)
def guidance_embedding(w, dim=256):
    """A full LCM is distilled with the guidance scale w as an input: (w − 1) · 1000 through sin and cos at dim/2
    frequencies (sin first). LCM-LoRA has no such input, since a LoRA can only change weights that exist."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000.0) / (half - 1) * torch.arange(half))
    angles = ((w - 1) * 1000.0)[:, None] * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class GuidedTimestepEmbedding(TimestepEmbedding):
    """A full LCM's timestep MLP: the guidance embedding, through a Linear without bias, is added to the
    timestep's sinusoid before the first layer."""

    def __init__(self, width: int, guidance_dim: int = 256):
        super().__init__(width)
        self.cond_proj = nn.Linear(guidance_dim, width, bias=False)

    def forward(self, t, w):
        x = sinusoid(t, self.width) + self.cond_proj(guidance_embedding(w, self.cond_proj.in_features))
        return self.linear_2(F.silu(self.linear_1(x)))


# %% [lcm-step] Consistency step
def lcm_timesteps(steps: int, c: Config):
    """The LoRA was distilled on the teacher's 50-step grid 999, 979, …, 19. Sampling takes \`steps\` of those,
    evenly spaced from the top: 999, 499 for 2 steps; 999, 759, 499, 259 for 4."""
    k = c.timesteps // c.original_steps
    grid = (torch.arange(1, c.original_steps + 1) * k - 1).flip(0)
    return grid[torch.arange(steps) * c.original_steps // steps]


def boundary(t, c: Config):
    """c_skip and c_out make the consistency function the identity at t = 0, f(x, 0) = x. With t scaled by 10
    they are ≈ 0 and ≈ 1 at every timestep the sampler visits, so f is the U-Net's x̂₀ there."""
    s = t * c.timestep_scaling
    return c.sigma_data**2 / (s**2 + c.sigma_data**2), s / (s**2 + c.sigma_data**2) ** 0.5


def lcm_step(z, eps, t, t_next, alphas_cumprod, c: Config, noise=None):
    """Jump from z_t to a clean latent; then, unless this was the last step, noise it to the next (lower)
    timestep with fresh noise, as the forward process would. Returns z for the next step and the clean latent."""
    a = alphas_cumprod[t]
    x0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
    c_skip, c_out = boundary(t, c)
    denoised = c_out * x0 + c_skip * z
    if noise is None:
        return denoised, denoised
    a_next = alphas_cumprod[t_next]
    return a_next.sqrt() * denoised + (1 - a_next).sqrt() * noise, denoised


# %% [guidance] Low or no classifier-free guidance
def lcm_eps(unet, z, t, prompt, empty=None, scale=1.0):
    """LCM-LoRA learned the teacher's guided predictions in distillation, so at scale 1 it runs one U-Net pass per
    step, with the prompt only. Scales up to about 2 follow the prompt more closely at twice the cost; larger
    ones overshoot."""
    if scale <= 1:
        return unet(z, t, prompt)
    eps_empty, eps_prompt = unet(torch.cat([z, z]), t.repeat(2), torch.cat([empty, prompt])).chunk(2)
    return guide(eps_empty, eps_prompt, scale)


# %% [model] Text to image in four steps
@torch.no_grad()
def sample(unet, vae, prompt, c: Config, generator, size=256, steps=4, empty=None, scale=1.0):
    """LCM-LoRA, already merged into \`unet\`: \`steps\` consistency steps, then the VAE decodes. \`prompt\` and \`empty\`
    are the CLIP text encoder's outputs, (1, 77, 768) each."""
    alphas_cumprod = noise_schedule(c)
    latent = size // 2 ** (len(c.vae_widths) - 1)
    z = torch.randn(1, c.latent_channels, latent, latent, generator=generator)
    timesteps = lcm_timesteps(steps, c)
    for i, t in enumerate(timesteps):
        eps = lcm_eps(unet, z, t.reshape(1), prompt, empty, scale)
        last = i == len(timesteps) - 1
        noise = None if last else torch.randn(z.shape, generator=generator)
        z, _ = lcm_step(z, eps, t, None if last else timesteps[i + 1], alphas_cumprod, c, noise)
    return (vae.decode(z / c.vae_scale) / 2 + 0.5).clamp(0, 1)
`;export{e as default};
