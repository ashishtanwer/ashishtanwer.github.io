# Recorded LCM-LoRA on Stable Diffusion 1.5

`manifest.json`, its `.f32` files and images are recorded from
[latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/tree/cf2fced511dbe7e26c8d1d397e728fbab875db4b)
(Luo et al., [arXiv:2311.05556](https://arxiv.org/abs/2311.05556); latent consistency models,
[arXiv:2310.04378](https://arxiv.org/abs/2310.04378)), revision
`cf2fced511dbe7e26c8d1d397e728fbab875db4b`, under the openrail++ license of its
[model card](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/blob/cf2fced511dbe7e26c8d1d397e728fbab875db4b/README.md),
on [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14),
revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, under the
[CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
The files hold activations, small corners of weights and generated images, never whole weights.
`scripts/diffusion/record_lcm.py` writes them through diffusers' own `StableDiffusionPipeline`
(safety checker not loaded), with the LoRA loaded by `load_lora_weights` and merged by `fuse_lora`;
hooks only observe, and an unhooked 4-step call gives the identical image.

## Runs

The SD 1.5 tab's prompt `a red apple on a wooden table` and `torch.Generator("cpu").manual_seed(0)`,
at 384 × 384 pixels (a 4 × 48 × 48 latent):

| Run          | U-Net         | Sampler                   | Guidance         |
| ------------ | ------------- | ------------------------- | ---------------- |
| `ddim-25`    | SD 1.5        | `DDIMScheduler`, 25 steps | 7.5 (two passes) |
| `ddim-4`     | SD 1.5        | `DDIMScheduler`, 4 steps  | 7.5              |
| `base-4`     | SD 1.5        | `LCMScheduler`, 4 steps   | none (one pass)  |
| `lcm-4`      | LoRA merged   | `LCMScheduler`, 4 steps   | none; in depth   |
| `lcm-2`      | LoRA merged   | `LCMScheduler`, 2 steps   | none             |
| `lcm-4-w*`   | LoRA merged   | `LCMScheduler`, 4 steps   | 1.5, 2 and 4     |
| `lcm-4-half` | LoRA at s 0.5 | `LCMScheduler`, 4 steps   | none             |

## Files

| File           | What it holds                                                                                                                                                                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lcm.f32`      | the 4-step run: zₜ, ε̂, x̂₀, the consistency output, the fresh noise, the timestep's sinusoid and embedding; SD 1.5's ε̂ and x̂₀ for the same zₜ; ε(∅), ε(prompt) and ε̂ of the guidance-1.5 run; the guidance-scale embedding diffusers would give a full LCM for w = 1, 1.5, 2, 4 |
| `lora.f32`     | for the attention projection the LoRA moves most: W, ΔW = (α / r)·B·A and W + ΔW (top-left 48 × 48), A and B (48 columns or rows), the singular values of W and ΔW; the same singular values for the 3 × 3 convolution it moves most; ‖ΔW‖ / ‖W‖ for all 278 layers            |
| `step-TTT.f32` | each of the four steps: every U-Net block's RMS map with the LoRA, without it on the same input, and the RMS of the difference                                                                                                                                                 |
| `*.png`        | each run's final image, as the pipeline returns it                                                                                                                                                                                                                             |
| `*-x0.jpg`     | 128 px JPEG decodes of each step's clean estimate (x̂₀, or the consistency output) per run, of zₜ for `lcm-4`, and of SD 1.5's x̂₀ on `lcm-4`'s zₜ, by the pipeline's VAE                                                                                                        |

Each LCM step is checked against the scheduler's rule applied to the recorded noise prediction, with
the fresh noise replayed from the same generator; the merged weights against W + s·ΔW from the LoRA
file in float64; the DDIM steps and guidance mixes as in the SD 1.5 tab. `validation` in the manifest
lists every check's largest error.
