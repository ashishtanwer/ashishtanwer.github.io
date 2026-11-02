const e=`# IP-Adapter (h94/IP-Adapter, models/ip-adapter_sd15.safetensors) on Stable Diffusion 1.5, the model of the IP-Adapter tab,
# as plain PyTorch: a photo is the prompt. CLIP's image encoder (OpenCLIP ViT-H/14, models/image_encoder) turns it into one
# 1024-wide embedding, a linear layer splits that into 4 tokens of the text's width, and every cross-attention of SD 1.5's
# frozen U-Net attends to those 4 tokens with keys and values of its own, next to the text's: 22 M new weights in all.
# The U-Net, its blocks, the VAE, guidance and the DDIM step are the SD 1.5 tab's (pytorch/diffusion_sd15.py).
# Each "# %% [id]" section is one step of the tab's diagram.
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_sd15 import Attention, UNet, ddim_step, ddim_timesteps, guide, noise_schedule
from diffusion_sd15 import Config as SD15Config


@dataclass
class Config(SD15Config):
    image_size: int = 224  # the image encoder's input: 16 × 16 patches of 14 px
    patch_size: int = 14
    vision_width: int = 1280  # OpenCLIP ViT-H/14
    vision_layers: int = 32
    vision_heads: int = 16  # 80 dimensions per head
    vision_mlp: int = 5120
    image_embed_dim: int = 1024  # the projected class token: CLIP's image embedding
    image_tokens: int = 4  # the embedding becomes 4 tokens as wide as the text's (768)
    ip_scale: float = 1.0  # the image branch's weight in every cross-attention; 0 is SD 1.5 alone
    clip_mean: tuple = (0.48145466, 0.4578275, 0.40821073)  # CLIP's per-channel pixel statistics
    clip_std: tuple = (0.26862954, 0.26130258, 0.27577711)


# %% [image-encoder] CLIP image encoder
def bicubic_weights(size_in, size_out):
    """PIL's bicubic resampling of one axis as a (size_out, size_in) matrix: Keys' cubic (a = −0.5), stretched by the
    shrink factor so every input pixel is averaged in (antialiasing), each row summing to 1, then in fixed point (× 2²²)."""
    scale = size_in / size_out
    center = (torch.arange(size_out, dtype=torch.float64) + 0.5) * scale  # each output pixel's centre in the input
    x = ((torch.arange(size_in, dtype=torch.float64)[None] - center[:, None] + 0.5) * (1 / max(scale, 1.0))).abs()
    w = torch.where(x < 1, (1.5 * x - 2.5) * x * x + 1, torch.where(x < 2, (((x - 5) * x + 8) * x - 4) * -0.5, 0.0))
    w = w / w.sum(1, keepdim=True)
    return torch.trunc(w * 2**22 + 0.5 * w.sign())  # rounded half away from zero


def resize(photo, height, width):
    """(3, H, W) values 0–255 → (3, height, width) as PIL resizes: every row first (the width), then every column, each
    pass in integers (+ ½, then ⌊· / 2²²⌋) clipped to 0–255."""
    one_axis = lambda x, weights: ((x @ weights.T + 2**21) / 2**22).floor().clamp(0, 255)
    rows = one_axis(photo.double(), bicubic_weights(photo.shape[2], width))  # (3, H, width)
    return one_axis(rows.transpose(1, 2), bicubic_weights(photo.shape[1], height)).transpose(1, 2)


def preprocess(photo, c: Config):
    """CLIPImageProcessor: the shortest side to 224 (bicubic, the long side rounded down), the centre 224 × 224, values
    0–255 → 0–1, then CLIP's mean and std. (3, H, W) → (1, 3, 224, 224)."""
    h, w = photo.shape[1:]
    long = int(c.image_size * max(h, w) / min(h, w))
    image = resize(photo, *((c.image_size, long) if h <= w else (long, c.image_size)))
    top, left = (image.shape[1] - c.image_size) // 2, (image.shape[2] - c.image_size) // 2
    image = (image[:, top : top + c.image_size, left : left + c.image_size] * (1 / 255)).float()
    mean, std = torch.tensor(c.clip_mean)[:, None, None], torch.tensor(c.clip_std)[:, None, None]
    return ((image - mean) / std)[None]


class VisionLayer(nn.Module):
    """Pre-LayerNorm transformer layer over the class token and the 256 patches; no mask, exact (erf) GELU."""

    def __init__(self, c: Config):
        super().__init__()
        d, self.heads = c.vision_width, c.vision_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.out = (nn.Linear(d, d) for _ in range(4))
        self.fc1, self.fc2 = nn.Linear(d, c.vision_mlp), nn.Linear(c.vision_mlp, d)

    def forward(self, x):  # x: (batch, 257, 1280)
        b, n, d = x.shape
        h = self.norm1(x)
        q, k, v = (proj(h).view(b, n, self.heads, -1).transpose(1, 2) for proj in (self.q, self.k, self.v))
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)
        x = x + self.out((weights @ v).transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class ImageEncoder(nn.Module):
    """Preprocessed image → 14 × 14-pixel patches, each a 1280-wide token, behind a learned class token → LayerNorm
    (CLIP's pre_layrnorm) → 32 layers → the class token, LayerNormed and projected: the 1024-wide image embedding."""

    def __init__(self, c: Config):
        super().__init__()
        d = c.vision_width
        self.patch_embedding = nn.Conv2d(3, d, c.patch_size, stride=c.patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(d))
        self.position_embedding = nn.Embedding((c.image_size // c.patch_size) ** 2 + 1, d)
        self.pre_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(VisionLayer(c) for _ in range(c.vision_layers))
        self.post_norm = nn.LayerNorm(d)
        self.projection = nn.Linear(d, c.image_embed_dim, bias=False)

    def forward(self, pixels):  # pixels: (batch, 3, 224, 224)
        patches = self.patch_embedding(pixels).flatten(2).transpose(1, 2)  # (batch, 256, 1280)
        x = torch.cat([self.class_embedding.expand(len(pixels), 1, -1), patches], dim=1) + self.position_embedding.weight
        x = self.pre_norm(x)
        for layer in self.layers:
            x = layer(x)
        return self.projection(self.post_norm(x[:, 0]))  # (batch, 1024)


# %% [image-projection] Image embedding → 4 image tokens
class ImageProjection(nn.Module):
    """CLIP's image embedding (1024) → one Linear to 4 × 768 → 4 tokens, each LayerNormed: what the U-Net reads of the
    photo. With the text's width, they can be attended to like words, but with keys and values of their own."""

    def __init__(self, c: Config):
        super().__init__()
        self.tokens = c.image_tokens
        self.proj = nn.Linear(c.image_embed_dim, c.image_tokens * c.text_dim)
        self.norm = nn.LayerNorm(c.text_dim)

    def forward(self, image_embeds):  # (batch, 1024) → (batch, 4, 768)
        return self.norm(self.proj(image_embeds).view(len(image_embeds), self.tokens, -1))


# %% [decoupled-attention] Decoupled cross-attention
class DecoupledCrossAttention(Attention):
    """SD 1.5's cross-attention (to_q, to_k, to_v, to_out, frozen) with a second key and value projection, to_k_ip and
    to_v_ip, for the image tokens: the adapter's only weights in the block. The same queries attend to the 77 text
    tokens and to the 4 image tokens in two separate softmaxes; the image's result, times \`scale\`, is added to the
    text's before to_out."""

    def __init__(self, dim, heads, context_dim):
        super().__init__(dim, heads, context_dim)
        self.to_k_ip = nn.Linear(context_dim, dim, bias=False)
        self.to_v_ip = nn.Linear(context_dim, dim, bias=False)

    def attend(self, x, text, image):
        """Both attention maps, (batch, heads, pixels, 77) and (batch, heads, pixels, 4), and what each one reads."""
        b, n, d = x.shape
        split = lambda t: t.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        merge = lambda t: t.transpose(1, 2).reshape(b, n, d)
        q = split(self.to_q(x))
        k_text, v_text = split(self.to_k(text)), split(self.to_v(text))
        k_image, v_image = split(self.to_k_ip(image)), split(self.to_v_ip(image))
        w_text = torch.softmax(q @ k_text.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)
        w_image = torch.softmax(q @ k_image.transpose(-1, -2) / math.sqrt(d // self.heads), dim=-1)
        return w_text, w_image, merge(w_text @ v_text), merge(w_image @ v_image)

    def forward(self, x, context):  # x: (batch, pixels, width); context: (text (batch, 77, 768), image (batch, 4, 768), scale)
        text, image, scale = context
        _, _, from_text, from_image = self.attend(x, text, image)
        return self.to_out(from_text + scale * from_image)


# %% [ip-unet] Image tokens in every cross-attention
class IPAdapterUNet(UNet):
    """SD 1.5's U-Net, frozen, with the image projection and every transformer block's cross-attention made decoupled:
    16 in SD 1.5 (6 down, 1 middle, 9 up), each with its own to_k_ip and to_v_ip, all reading the same 4 image tokens.
    The forward pass is SD 1.5's; the context it hands each block is (text, image tokens, scale) instead of the text."""

    def __init__(self, c: Config):
        super().__init__(c)
        self.image_proj = ImageProjection(c)
        for block in self.transformer_blocks():
            block.cross_attention = DecoupledCrossAttention(block.proj_in.in_channels, c.heads, c.text_dim)

    def transformer_blocks(self):
        """In the checkpoint's order: down, up, then the middle."""
        return [block for level in [*self.down, *self.up, self.mid] for block in level.attentions]

    def forward(self, z, t, text, image_embeds, scale=1.0):  # image_embeds: (batch, 1024), CLIP's
        return super().forward(z, t, (text, self.image_proj(image_embeds), scale))


@torch.no_grad()
def load_ip_adapter(unet: IPAdapterUNet, state):
    """ip-adapter_sd15.safetensors' tensors into the U-Net: image_proj.* is the projection, and ip_adapter.{1, 3, …, 31}
    are the 16 cross-attentions' to_k_ip / to_v_ip in transformer_blocks()' order (the even numbers were self-attentions)."""
    unet.image_proj.load_state_dict({k.removeprefix("image_proj."): v for k, v in state.items() if k.startswith("image_proj.")})
    for i, block in enumerate(unet.transformer_blocks()):
        for name in ("to_k_ip", "to_v_ip"):
            getattr(block.cross_attention, name).weight.copy_(state[f"ip_adapter.{2 * i + 1}.{name}.weight"])


# %% [guidance] Guidance with an image prompt
def guided_eps(unet, z, t, empty, prompt, image_embeds, scale=1.0, guidance=7.5):
    """Classifier-free guidance, as in SD 1.5. The empty-prompt pass also gets an empty image: an all-zero CLIP embedding,
    whose 4 tokens are then LayerNorm(the projection's bias), not zeros. Guidance amplifies text and image together."""
    text = torch.cat([empty, prompt])
    images = torch.cat([torch.zeros_like(image_embeds), image_embeds])
    eps_empty, eps_prompt = unet(torch.cat([z, z]), t.repeat(2), text, images, scale).chunk(2)
    return guide(eps_empty, eps_prompt, guidance)


# %% [model] Image prompt + text prompt → image
@torch.no_grad()
def sample(unet, encoder, vae, photo, prompt, empty, c: Config, generator, size=256, steps=20, scale=None, guidance=7.5):
    """The CLIP image encoder embeds \`photo\` (3 × H × W, values 0–255) once; DDIM with guidance then denoises with
    the image prompt at \`scale\` (the config's 1.0 by default) and the VAE decodes the result. \`prompt\` and \`empty\` are
    the CLIP text encoder's outputs, (1, 77, 768) each."""
    scale = c.ip_scale if scale is None else scale
    image_embeds = encoder(preprocess(photo, c))  # (1, 1024)
    alphas_cumprod = noise_schedule(c)
    latent = size // 2 ** (len(c.vae_widths) - 1)
    z = torch.randn(1, c.latent_channels, latent, latent, generator=generator)
    for t in ddim_timesteps(steps, c):
        z, _ = ddim_step(z, guided_eps(unet, z, t, empty, prompt, image_embeds, scale, guidance), t, alphas_cumprod, steps)
    return (vae.decode(z / c.vae_scale) / 2 + 0.5).clamp(0, 1)
`;export{e as default};
