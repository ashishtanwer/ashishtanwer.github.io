# Recorded Stable Diffusion 1.5 inpainting

`manifest.json`, its `.f32` files and images are recorded from the pretrained
[stable-diffusion-v1-5/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/tree/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb)
checkpoint (Rombach et al., [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)), revision
`8a4288a76071f7280aedbdb3253bdb9e9d5d84bb`, and, for the comparison,
[stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14),
both under the [CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
The inpainting repo's VAE and text encoder are byte-identical to SD 1.5's (the Hub lists the same SHA-256),
so SD 1.5's safetensors copies of them are loaded. The files hold activations and generated images, never
the weights. `scripts/diffusion/record_inpainting.py` writes them through diffusers' own
`StableDiffusionInpaintPipeline` and `StableDiffusionImg2ImgPipeline` (safety checker not loaded); hooks
only observe, and unhooked calls give identical images.

## Runs

The photo below at 384 × 384, SAM's mask of the dog, one prompt (`prompt` in the manifest), the empty prompt
for classifier-free guidance at scale 7.5, 20 DDIM steps (the inpainting repo's scheduler) from pure noise
and `torch.Generator("cpu").manual_seed(0)`. Then SD 1.5's own img2img with the same photo, prompt and seed
at strength 0.8, with the photo pasted back outside the mask.

The mask is the SAM tab's (`static/traces/sam/prompt-1-mask-1.png`, SAM ViT-B, the box around the dog):
the centre square, nearest-neighbour to 384 × 384, then widened by 4 px on every side.

## Files

| File                                  | What it holds                                                                                                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inputs.f32`                          | the mask at latent size and the masked photo's latent as the U-Net reads them, the whole photo's latent img2img starts from, and ‖W‖ of conv_in over its 9 input channels |
| `trajectory.f32`                      | at every step: z_t, ε(∅), ε(prompt) and the guided ε̂                                                                                                                      |
| `step-TTT.f32`                        | steps t = 901, 501, 101, prompt pass: every block's RMS over channels, and conv_in's output split into the noisy latent's, the mask's and the masked latent's parts       |
| `photo.jpg`, `mask.png`, `masked.jpg` | the photo, the mask (white is repainted) and the photo with the hole at 0 in [−1, 1]                                                                                      |
| `masked-decoded.jpg`                  | the masked latent decoded by the VAE                                                                                                                                      |
| `result.png`                          | the inpainting pipeline's image                                                                                                                                           |
| `img2img.jpg`, `pasted.png`           | SD 1.5 img2img's image, and the same with the photo outside the mask                                                                                                      |
| `x0-*.jpg`, `xt.jpg`                  | 128 px JPEG decodes of every step's x̂₀ (both runs) and of the inpainting run's z_t, by the pipeline's VAE                                                                 |

The photo is [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg) by Jon Sullivan, public
domain, from `static/architectures/sam-beagle.jpg`: the centre square, resized with Lanczos.

Each DDIM step is checked against the update applied to the recorded ε̂, the guidance against the two
recorded predictions, the mask channel against nearest-neighbour downsampling, the masked latent against the
VAE's mean and standard deviation with the recorded noise, both noises against the seeded generator, and
conv_in's parts against its output; `validation` in the manifest lists every check's largest error.
