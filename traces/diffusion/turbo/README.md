# Recorded SDXL-Turbo

`manifest.json`, its `.f32` files and images are recorded from the pretrained
[stabilityai/sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo/tree/71153311d3dbb46851df1931d3ca6e939de83304)
checkpoint (Sauer et al., Adversarial Diffusion Distillation,
[arXiv:2311.17042](https://arxiv.org/abs/2311.17042)), revision
`71153311d3dbb46851df1931d3ca6e939de83304`, fp16 weights upcast to float32, under the
[Stability AI Non-Commercial Research Community License](https://huggingface.co/stabilityai/sdxl-turbo/blob/71153311d3dbb46851df1931d3ca6e939de83304/LICENSE.md).
`baseline.json`, `baseline.f32` and `baseline-*.jpg` are recorded from
[stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b)
at revision `462165984030d82259a11f4367a4eed129e94a7b` (CreativeML Open RAIL++-M). The files hold
activations, generated images and weight statistics, never the weights.

`scripts/diffusion/record_turbo.py` writes them through diffusers' own `StableDiffusionXLPipeline`
(no watermarker); hooks only observe, and unhooked calls give identical images.

## Runs

The prompt `a red apple on a wooden table`, 512 × 512 pixels (a 4 × 64 × 64 latent), guidance 0
(one U-Net pass per step, on the prompt alone) and `torch.Generator("cpu").manual_seed(0)`, with
SDXL-Turbo's own `EulerAncestralDiscreteScheduler` at trailing timesteps:

| Run       | Weights                  | Steps | Timesteps          |
| --------- | ------------------------ | ----- | ------------------ |
| `turbo-1` | SDXL-Turbo               | 1     | 999                |
| `turbo-4` | SDXL-Turbo               | 4     | 999, 749, 499, 249 |
| `base-1`  | SDXL base (`--baseline`) | 1     | 999                |
| `base-4`  | SDXL base (`--baseline`) | 4     | 999, 749, 499, 249 |

The four runs share the initial noise, and the one-step runs give both U-Nets the identical input.

## Files

| File                               | What it holds                                                                                                                                                                                                                                                                                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text.f32`                         | both encoders' second-to-last layer for the prompt (first 8 numbers and norms at all 77 positions) and bigG's pooled vector                                                                                                                                                                                                                               |
| `conditioning.f32`                 | the six micro-conditioning numbers, their sinusoids and the added embedding; the timestep's sinusoid and temb at each of the five U-Net calls                                                                                                                                                                                                             |
| `onestep.f32`                      | the one-step call in depth: z_T, the U-Net's input, ε̂ and x̂₀; every block's RMS over channels; `down.2.resnet.0` stage by stage; the 10-layer `up.0.transformer.0` layer by layer with all 20 heads' attention of layers 1 and 10 for the latent's middle row; every transformer block's cross-attention averaged over heads and layers, per prompt token |
| `fewstep.f32`                      | the four-step run: z_t before every step and the final latent, ε̂, x̂₀, the fresh noise each step draws, and every block's RMS map at every step                                                                                                                                                                                                            |
| `turbo-*-final.jpg`                | both Turbo runs' images (JPEG, quality 92)                                                                                                                                                                                                                                                                                                                |
| `turbo-*-x0.jpg`, `turbo-*-xt.jpg` | 128 px JPEG decodes of every step's x̂₀ and z_t                                                                                                                                                                                                                                                                                                            |
| `baseline.json`                    | the SDXL base runs' provenance and checks, and the weight comparison: ‖W_turbo − W_base‖ / ‖W_base‖ for the U-Net, every drawn block and every transformer layer, and for the VAE                                                                                                                                                                         |
| `baseline.f32`                     | SDXL base on the one-step input: z_T, the U-Net's input, ε̂, x̂₀, its timestep and added embeddings, and every block's RMS map                                                                                                                                                                                                                              |
| `baseline-*.jpg`                   | the SDXL base runs' images and x̂₀ decodes                                                                                                                                                                                                                                                                                                                 |
