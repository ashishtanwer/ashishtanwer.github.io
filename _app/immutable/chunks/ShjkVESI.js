const e=`# Few-step distillation, the Diffusion Explainer's Few-step tab, as plain PyTorch: the step each few-step model takes
# (LCM-LoRA, SDXL-Turbo, the ImageNet-64 consistency model), the many-step sampler of the model each was distilled from,
# and how the page scores their images. The networks are the member tabs' (pytorch/diffusion_lcm.py,
# diffusion_turbo.py, diffusion_consistency.py, with SD 1.5's and SDXL's U-Nets); here \`unet\` is any callable.
# Each "# %% [id]" section is one panel of the tab.
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Config:
    timesteps: int = 1000  # SD 1.5 and SDXL: DDPM's 1000 timesteps, "scaled linear" betas
    beta_start: float = 0.00085
    beta_end: float = 0.012
    original_steps: int = 50  # the teacher's DDIM grid LCM-LoRA was distilled on: 999, 979, …, 19
    timestep_scaling: float = 10.0
    sigma_data: float = 0.5  # the standard deviation of the images (all three use 0.5)
    sigma_min: float = 0.002  # the consistency model and its EDM teacher: σ from 80 down to ε = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    levels: int = 40  # the noise levels the consistency model was distilled on, and the teacher's Heun steps


def noise_schedule(c: Config):
    """"Scaled linear": √β grows linearly, from √0.00085 to √0.012 (SD 1.5's and SDXL's training schedule)."""
    betas = torch.linspace(c.beta_start**0.5, c.beta_end**0.5, c.timesteps) ** 2
    return torch.cumprod(1 - betas, dim=0)  # ᾱ_t


# %% [ddim-step] SD 1.5's DDIM step: LCM's teacher
def ddim_timesteps(steps, c: Config):
    """Every 1000/steps-th timestep, offset by one: 961, 921, …, 1 for 25 steps."""
    return torch.arange(0, steps).flip(0) * (c.timesteps // steps) + 1


def ddim_step(z, eps, t, alphas_cumprod, steps):
    """Deterministic (η = 0): estimate the clean latent x̂₀, then re-noise it to the previous timestep with the same
    ε̂. With guidance w, ε̂ = ε(∅) + w · (ε(prompt) − ε(∅)): two U-Net passes per step, 50 for 25 steps."""
    a_t = alphas_cumprod[t]
    prev = t - len(alphas_cumprod) // steps
    a_prev = alphas_cumprod[prev] if prev >= 0 else alphas_cumprod[0]
    x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
    return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps, x0


# %% [lcm-step] LCM: the consistency step
def lcm_timesteps(steps: int, c: Config):
    """Steps taken from the teacher's 50-step grid, evenly from the top: 999; 999, 499; 999, 759, 499, 259."""
    k = c.timesteps // c.original_steps
    grid = (torch.arange(1, c.original_steps + 1) * k - 1).flip(0)
    return grid[torch.arange(steps) * c.original_steps // steps]


def boundary(t, c: Config):
    """c_skip and c_out make the consistency function the identity at t = 0. With t scaled by 10 they are ≈ 0 and ≈ 1
    at every timestep the sampler visits, so f is the U-Net's own x̂₀ there."""
    s = t * c.timestep_scaling
    return c.sigma_data**2 / (s**2 + c.sigma_data**2), s / (s**2 + c.sigma_data**2) ** 0.5


def lcm_step(z, eps, t, t_next, alphas_cumprod, c: Config, noise=None):
    """Latent consistency distillation trained f(z_t, t) to agree with f one teacher DDIM step further down the same
    guided trajectory, so f jumps from any z_t to that trajectory's end. Sampling: jump, then (unless this was the
    last step) noise the clean latent back to the next timestep with fresh noise."""
    a = alphas_cumprod[t]
    x0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
    c_skip, c_out = boundary(t, c)
    denoised = c_out * x0 + c_skip * z
    if noise is None:
        return denoised, denoised
    a_next = alphas_cumprod[t_next]
    return a_next.sqrt() * denoised + (1 - a_next).sqrt() * noise, denoised


# %% [euler-step] SDXL base's Euler step: Turbo's teacher
def euler_schedule(steps, c: Config):
    """Every 1000/steps-th timestep, offset by one (961, 921, …, 1 for 25 steps), and its noise level
    σ = √((1 − ᾱ) / ᾱ), then σ = 0. The run starts at z = √(σ₀² + 1) · noise."""
    alphas_cumprod = noise_schedule(c)
    timesteps = torch.arange(steps).flip(0) * (c.timesteps // steps) + 1
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    return timesteps, torch.cat([sigmas[timesteps], torch.zeros(1)])


def euler_step(z, eps, sigma, sigma_next):
    """Here z = x_0 + σ · ε (the U-Net reads z / √(σ² + 1)). One Euler step of dz/dσ = ε̂ from σ to σ_next: the
    probability-flow ODE, so the run is deterministic once the first noise is drawn."""
    x0 = z - sigma * eps
    return z + (sigma_next - sigma) * (z - x0) / sigma, x0


# %% [ancestral-step] Turbo: Euler ancestral step
def trailing_schedule(steps, c: Config):
    """Timesteps spaced back from the very last one: 999; 999, 499; 999, 749, 499, 249; each one's σ, then 0.
    Turbo was trained on exactly these four timesteps, starting from pure noise at t = 999."""
    alphas_cumprod = noise_schedule(c)
    timesteps = (torch.arange(c.timesteps, 0, -c.timesteps / steps).round() - 1).long()
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    return timesteps, torch.cat([sigmas[timesteps], torch.zeros(1)])


def ancestral_step(z, eps, sigma, sigma_next, noise):
    """Adversarial diffusion distillation trained x̂₀ = z − σ·ε̂ to fool a discriminator and to score well under the
    teacher, at any of the four timesteps; nothing ties it to the teacher's trajectory. Step down to σ_down with
    Euler, then add fresh noise σ_up so that σ_up² + σ_down² = σ_next². The last step lands on x̂₀."""
    x0 = z - sigma * eps
    sigma_up = (sigma_next**2 * (sigma**2 - sigma_next**2) / sigma**2) ** 0.5
    sigma_down = (sigma_next**2 - sigma_up**2) ** 0.5
    return z + (sigma_down - sigma) * (z - x0) / sigma + noise * sigma_up, x0


# %% [heun] The EDM teacher: Heun on the probability-flow ODE
def edm_levels(c: Config):
    """The 40 noise levels from σ = 80 down to 0.002, evenly spaced in σ^(1/ρ) (Karras et al.): the grid the
    consistency model was distilled on, and the teacher's steps here."""
    ramp = torch.arange(c.levels, dtype=torch.float64) / (c.levels - 1)
    lo, hi = c.sigma_min ** (1 / c.rho), c.sigma_max ** (1 / c.rho)
    return ((hi + ramp * (lo - hi)) ** c.rho).float()


def edm_denoiser(unet, x, sigma, labels, c: Config):
    """The teacher's D(x; σ): EDM's scalings c_skip = σd² / (σ² + σd²), c_out = σ·σd / √(σ² + σd²) and
    c_in = 1 / √(σ² + σd²), with σ fed to the U-Net as 250 · ln σ (as the consistency models code does), clipped."""
    c_skip = c.sigma_data**2 / (sigma**2 + c.sigma_data**2)
    c_out = sigma * c.sigma_data / (sigma**2 + c.sigma_data**2) ** 0.5
    out = unet(x / (sigma**2 + c.sigma_data**2) ** 0.5, 250 * torch.log(torch.as_tensor(sigma, dtype=torch.float64)), labels)
    return (c_skip * x + c_out * out).clamp(-1, 1)


def heun_sample(denoise, x, sigmas):
    """Heun's method on dx/dσ = (x − D(x; σ)) / σ, deterministic: an Euler step, then the slope at its end averaged
    in. The last step, to σ = 0, is Euler alone and lands on D(x; σ_min): 2 · 40 − 1 = 79 calls. Returns the image
    and the path, the x the teacher held before each step with its D(x; σ)."""
    sigmas = torch.cat([sigmas, torch.zeros(1)])
    path = []
    for i in range(len(sigmas) - 1):
        denoised = denoise(x, sigmas[i])
        path.append((x, denoised))
        d = (x - denoised) / sigmas[i]
        dt = sigmas[i + 1] - sigmas[i]
        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x2 = x + d * dt
            x = x + (d + (x2 - denoise(x2, sigmas[i + 1])) / sigmas[i + 1]) / 2 * dt
    return x, path


# %% [consistency-function] The consistency model: f(x, σ)
def consistency_function(unet, x, sigma, labels, c: Config):
    """Consistency distillation trained f on pairs of points one teacher Heun step apart on the same ODE path, to give
    the same output at both; with f(x, ε) = x, every point of a path maps to its start. One U-Net call, clipped.
    c_skip and c_out are EDM's, shifted by ε so that c_skip(ε) = 1 and c_out(ε) = 0."""
    s = sigma.float()  # the scheduler keeps σ in float32 …
    c_skip = c.sigma_data**2 / ((s - c.sigma_min) ** 2 + c.sigma_data**2)
    c_out = (s - c.sigma_min) * c.sigma_data / (s**2 + c.sigma_data**2) ** 0.5
    out = unet(x / (s**2 + c.sigma_data**2) ** 0.5, 250 * torch.log(sigma + 1e-44), labels)  # … but ln σ in float64
    return (c_skip * x + c_out * out).clamp(-1, 1)


# %% [multistep] Multistep: renoise, jump again
def renoise(x0, sigma_next, z, c: Config):
    """Back up to a lower noise level with fresh noise, x = x̂₀ + √(σ² − ε²) · z, and apply f again. Each step draws
    new noise, so more steps give a different image, not a sharper copy of the one-step image."""
    return x0 + z * (sigma_next**2 - c.sigma_min**2).sqrt()


def cm_schedule(timesteps, c: Config):
    """Each call's σ, as CMStochasticIterativeScheduler reads the list: positions on the 40 levels, read backwards.
    [0] is σ = 80; [22, 0] is σ = 80 then ≈ 1.38 (the paper's two steps); 4 steps take positions 0, 10, 20, 30."""
    ramp = torch.tensor(timesteps[::-1], dtype=torch.float64) / (c.levels - 1)
    lo, hi = c.sigma_min ** (1 / c.rho), c.sigma_max ** (1 / c.rho)
    return (hi + ramp * (lo - hi)) ** c.rho


# %% [scores] CLIP score and distance to the base model's image
def clip_score(image_embedding, text_embedding):
    """CLIP maps an image and a text into one space; the score is 100 × their cosine, floored at 0 (as torchmetrics'
    CLIPScore). It measures how well an image matches its text, not how good it looks."""
    return 100 * F.cosine_similarity(image_embedding, text_embedding, dim=-1).clamp(min=0)


def distance(image, reference):
    """Pixel RMS distance of two images in [0, 1]: 0 when a few-step run lands where its base model's run from the
    same noise ends."""
    return ((image - reference) ** 2).mean().sqrt()
`;export{e as default};
