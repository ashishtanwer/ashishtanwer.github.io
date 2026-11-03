# Recorded few-step models beside their base models

`manifest.json` and the sprites are recorded by `scripts/diffusion/record_few_step.py` from these checkpoints, at
pinned revisions, through diffusers' own pipelines; hooks and wrapped scheduler steps only observe, and unobserved
calls give identical images. The files hold generated images (64 px decodes of every step, 160 px finals) and
numbers, never weights.

| Member      | Few-step model                                                                                                                                                                                                                                                                                      | Base model                                                                                                                                                                                                                                                           |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| lcm         | [latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/tree/cf2fced511dbe7e26c8d1d397e728fbab875db4b) @ `cf2fced`, openrail++                                                                                                                               | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14) @ `451f4fe`, [CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) |
| turbo       | [stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo/tree/71153311d3dbb46851df1931d3ca6e939de83304) @ `7115331`, [Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/71153311d3dbb46851df1931d3ca6e939de83304/LICENSE.md) | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b) @ `4621659`, CreativeML Open RAIL++-M                                                                      |
| consistency | [openai/diffusers-cd_imagenet64_l2](https://huggingface.co/openai/diffusers-cd_imagenet64_l2/tree/5f462e4403fc37b72ec6004e806c71805db22387) @ `5f462e4`, MIT                                                                                                                                        | [dg845/diffusers-cm_edm_imagenet64_ema](https://huggingface.co/dg845/diffusers-cm_edm_imagenet64_ema/tree/291b33e828facdff9960eabf73b9adbcec1210e4) @ `291b33e`, MIT: OpenAI's `edm_imagenet64_ema` teacher as a diffusers UNet2DModel                               |

CLIP scores use [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41)
@ `32bd642`.

## Runs

Seeds 0–7 (`torch.Generator("cpu").manual_seed(seed)`), the prompt `a red apple on a wooden table` (the consistency
model takes ImageNet class 948, Granny Smith), 1, 2 and 4 steps of each few-step model, and each base model's
many-step run from the same unit noise:

| Member      | Size   | Few-step sampler                                      | Base run                                                       |
| ----------- | ------ | ----------------------------------------------------- | -------------------------------------------------------------- |
| lcm         | 384 px | LCMScheduler, no guidance                             | DDIM, 25 steps, guidance 7.5 (50 U-Net calls)                  |
| turbo       | 512 px | Euler ancestral at trailing timesteps, guidance 0     | SDXL's Euler, 25 steps, guidance 5 (50 calls)                  |
| consistency | 64 px  | CMStochasticIterativeScheduler: [0], [22, 0], 4 steps | Heun on the 40 Karras levels, deterministic (79 calls), on CPU |

From the base run's input before every step, each few-step model also makes its one-step jump; the manifest keeps the
pixel RMS distance of every jump, every step's x̂₀ and every final image to the base run's image. Seed 0 reproduces
the LCM tab's `ddim-25`, `lcm-2` and `lcm-4` images and the Turbo tab's one-step x̂₀ and four-step latents exactly
(`reproduced` in the manifest); `validation` lists every check's largest error.
