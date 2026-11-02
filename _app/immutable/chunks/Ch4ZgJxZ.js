const e=`# Stable Diffusion 1.5 inpainting (stable-diffusion-v1-5/stable-diffusion-inpainting), the model of the inpainting tab, as
# plain PyTorch: SD 1.5's U-Net design with five more input channels (trained from SD 1.2's weights), the mask and the VAE
# latent of the photo with the hole blanked out, so at every step it sees what it has to fit. The tab sets it beside SD 1.5
# img2img, which redraws the whole photo and then has the original pasted back outside the mask.
# The U-Net, its blocks, the VAE, guidance and the DDIM step are the SD 1.5 tab's (pytorch/diffusion_sd15.py).
# Each "# %% [id]" section is one step of the tab's diagram.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_sd15 import Config as SD15Config
from diffusion_sd15 import UNet, ddim_step, ddim_timesteps, guide, noise_schedule


@dataclass
class Config(SD15Config):
    in_channels: int = 9  # the U-Net's input: noisy latent 4 + mask 1 + masked-photo latent 4; it still predicts 4


# %% [mask] The mask, in pixels and in latent space
def prepare_mask(photo, mask, c: Config):
    """photo: (3, H, W) and mask: (H, W) or (3, H, W), values 0–255, both already at the output size; white in the
    mask is what gets repainted. → the binary mask (1, 1, H, W), the photo with the hole blanked out (1, 3, H, W) in
    [-1, 1], and the mask at latent size (1, 1, H/8, W/8)."""
    if mask.dim() == 3:  # an RGB mask → grey, rounded as PIL's convert("L") rounds it
        mask = (mask[0].long() * 19595 + mask[1].long() * 38470 + mask[2].long() * 7471 + 0x8000) >> 16
    mask = (mask.float() / 255 >= 0.5).float()[None, None]  # 128 and up repaints, 127 and below is kept
    image = photo.float()[None] / 255 * 2 - 1
    masked_image = image * (mask < 0.5)  # the hole becomes 0: mid-grey, not black
    f = 2 ** (len(c.vae_widths) - 1)  # 8
    h, w = mask.shape[-2:]
    mask_latent = F.interpolate(mask, size=(h // f, w // f), mode="nearest")  # the top-left pixel of each 8 × 8 block
    return mask, masked_image, mask_latent


# %% [masked-latent] Masked photo → latent
def encode(vae, image, c: Config, generator):
    """(1, 3, H, W) in [-1, 1] → (1, 4, H/8, W/8): a sample of the VAE's latent distribution, drawn with the
    generator (not its mean), × 0.18215, the same scale as the noisy latent it sits beside."""
    mean, logvar = vae.encode(image)
    return c.vae_scale * (mean + (0.5 * logvar).exp() * torch.randn(mean.shape, generator=generator))


# %% [unet-input] Nine input channels
class InpaintUNet(UNet):
    """SD 1.5's U-Net with a wider first convolution: the noisy latent, the mask and the masked photo's latent are
    stacked on channels, and conv_in takes the 9 to 320. Everything after it is the SD 1.5 tab's U-Net."""

    def __init__(self, c: Config):
        super().__init__(c)
        self.conv_in = nn.Conv2d(c.in_channels, c.widths[0], 3, padding=1)  # 9 → 320

    def forward(self, z, t, text, mask, masked_latent):  # z, masked_latent: (b, 4, h, w); mask: (b, 1, h, w)
        return super().forward(torch.cat([z, mask, masked_latent], dim=1), t, text)  # ε̂: (b, 4, h, w)


# %% [guidance] Guidance with the mask in both passes
def guided_eps(unet, z, t, empty, prompt, mask, masked_latent, guidance=7.5):
    """Classifier-free guidance, as in SD 1.5. The empty-prompt and the prompt pass both get the same mask and masked
    latent, so the difference that guidance amplifies is the prompt's alone."""
    text = torch.cat([empty, prompt])
    eps_empty, eps_prompt = unet(torch.cat([z, z]), t.repeat(2), text, torch.cat([mask, mask]),
                                 torch.cat([masked_latent, masked_latent])).chunk(2)
    return guide(eps_empty, eps_prompt, guidance)


# %% [img2img] SD 1.5 img2img, pasted back
def img2img_timesteps(steps, strength, c: Config):
    """The last int(steps × strength) of DDIM's timesteps: strength 0.8 of 20 steps starts at 751 and runs 16."""
    return ddim_timesteps(steps, c)[steps - min(int(steps * strength), steps):]


@torch.no_grad()
def img2img(unet, vae, photo, prompt, empty, c: Config, generator, steps=20, strength=0.8, guidance=7.5):
    """SD 1.5's own 4-channel U-Net started from the photo: its latent, noised to the first kept timestep, is
    denoised with guidance. Nothing tells it where the mask is, so it redraws the whole image."""
    latent = encode(vae, photo.float()[None] / 255 * 2 - 1, c, generator)  # the generator's first draw
    timesteps = img2img_timesteps(steps, strength, c)
    alphas_cumprod = noise_schedule(c)
    a = alphas_cumprod[timesteps[0]]
    z = a.sqrt() * latent + (1 - a).sqrt() * torch.randn(latent.shape, generator=generator)  # its second: the noise
    text = torch.cat([empty, prompt])
    for t in timesteps:
        eps_empty, eps_prompt = unet(torch.cat([z, z]), t.repeat(2), text).chunk(2)
        z, _ = ddim_step(z, guide(eps_empty, eps_prompt, guidance), t, alphas_cumprod, steps)  # steps: still 20
    return (vae.decode(z / c.vae_scale) / 2 + 0.5).clamp(0, 1)  # (1, 3, H, W) in [0, 1]


def paste(generated, photo, mask):
    """The generated pixels inside the binary mask, the photo's own outside it: (1, 3, H, W) in [0, 1]."""
    return generated * mask + photo.float()[None] / 255 * (1 - mask)


# %% [model] Photo + mask + prompt → image
@torch.no_grad()
def sample(unet, vae, photo, mask, prompt, empty, c: Config, generator, steps=20, guidance=7.5):
    """Inpainting at strength 1: DDIM from pure noise with guidance, the mask and the masked photo's latent beside z
    at every step; the VAE decodes the result. \`prompt\` and \`empty\` are the CLIP text encoder's outputs."""
    _, masked_image, mask_latent = prepare_mask(photo, mask, c)
    alphas_cumprod = noise_schedule(c)
    z = torch.randn(1, c.latent_channels, *mask_latent.shape[-2:], generator=generator)  # the generator's first draw
    masked_latent = encode(vae, masked_image, c, generator)  # its second
    for t in ddim_timesteps(steps, c):
        eps = guided_eps(unet, z, t, empty, prompt, mask_latent, masked_latent, guidance)
        z, _ = ddim_step(z, eps, t, alphas_cumprod, steps)
    # Nothing is pasted back: outside the mask too, every pixel is the VAE's redraw of the latent.
    return (vae.decode(z / c.vae_scale) / 2 + 0.5).clamp(0, 1)
`;export{e as default};
