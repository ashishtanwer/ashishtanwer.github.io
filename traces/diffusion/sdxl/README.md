# Recorded Stable Diffusion XL

`manifest.json`, its `.f32` files and images are recorded from the pretrained
[stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b)
checkpoint (Podell et al., [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)), revision
`462165984030d82259a11f4367a4eed129e94a7b`, fp16 weights upcast to float32, under the
[CreativeML Open RAIL++-M license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md).
The files hold activations and generated images, never the weights.
`scripts/diffusion/record_sdxl.py` writes them through diffusers' own `StableDiffusionXLPipeline`
(no watermarker); hooks only observe, and an unhooked call of the default run gives the identical
image.

## Runs

The prompt `a red apple on a wooden table`, no negative prompt (SDXL then zeroes the ∅ half's text
instead of encoding an empty string), guidance 5, 768 × 768 pixels (a 4 × 96 × 96 latent), 20 steps
of the model's own `EulerDiscreteScheduler` and `torch.Generator("cpu").manual_seed(0)`, four times
with different micro-conditioning (original size, crop top-left, target size). 768 is the smallest
size tried at which SDXL base draws a clean image: at 512 × 512 every run is covered in stripes.

| Run            | Original size | Crop     | Target size |
| -------------- | ------------- | -------- | ----------- |
| `default`      | 768 × 768     | 0, 0     | 768 × 768   |
| `original256`  | 256 × 256     | 0, 0     | 768 × 768   |
| `original1024` | 1024 × 1024   | 0, 0     | 768 × 768   |
| `crop256`      | 768 × 768     | 256, 256 | 768 × 768   |

## Files

| File                              | What it holds                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `text.f32`                        | both encoders' second-to-last layer for the prompt at all 77 positions: the first 8 of 768 (CLIP ViT-L/14) and of 1280 (OpenCLIP ViT-bigG/14) numbers and both norms; bigG's pooled vector (1280)                                                                                                                                                                                                                                                                                                                        |
| `conditioning.f32`                | per run: the six micro-conditioning numbers, their 256-wide sinusoids, and the added embedding of the prompt half and of the ∅ half                                                                                                                                                                                                                                                                                                                                                                                      |
| `trajectory.f32`                  | the default run's guided ε̂ at every step, averaged over 2 × 2 latent pixels (4 × 48 × 48) to fit                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `step-TTT.f32`                    | steps t = 701 and 301 of the default run: ε(∅), ε(prompt) and the guided ε̂ in full; the timestep's sinusoid, temb and temb + added; every block's RMS over channels (prompt half); `down.2.resnet.0` stage by stage; the 10-layer `up.0.transformer.0` layer by layer, with all 20 heads' self- and cross-attention of layers 1 and 10 for every third pixel (8) of the 24 × 24 level's middle row; every transformer block's cross-attention averaged over heads and layers, per prompt token, plus the padding's total |
| `vae.f32`                         | the VAE on a 512 × 512 photo: the latent distribution's mean and standard deviation, and mean × 0.13025                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `*-final.jpg`                     | each run's image, as the pipeline returns it (JPEG, quality 90)                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `*-x0.jpg`, `default-xt.jpg`      | 128 px JPEG decodes of every step's x̂₀ (and of the default run's z_t), by the pipeline's VAE                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `vae-input.jpg`, `vae-output.jpg` | the photo (Dog beagle, Jon Sullivan, public domain) and its reconstruction                                                                                                                                                                                                                                                                                                                                                                                                                                               |

`manifest.json` stores every self-check of the recording under `validation`: the Euler steps,
guidance, text and size embeddings, ResNet and every transformer layer rebuilt in float64 from the
captures, and the tab's PyTorch blocks (`pytorch/diffusion_sdxl.py`) run on recorded inputs with the
model's weights.
