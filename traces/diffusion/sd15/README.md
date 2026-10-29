# Recorded Stable Diffusion 1.5

`manifest.json`, its `.f32` files and images are recorded from the pretrained
[stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14)
checkpoint (Rombach et al., [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)), revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, under the
[CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
The files hold activations and generated images, never the weights.
`scripts/diffusion/record_sd15.py` writes them through diffusers' own `StableDiffusionPipeline`
(safety checker not loaded); hooks only observe, and an unhooked DDIM call gives the identical image.

## Runs

The prompt `a red apple on a wooden table`, the empty prompt for classifier-free guidance at
scale 7.5, 256 × 256 pixels (a 4 × 32 × 32 latent), 20 steps and
`torch.Generator("cpu").manual_seed(0)`, under three schedulers built from the model's own
scheduler config: `DDIMScheduler` (η = 0), `EulerDiscreteScheduler` and
`DPMSolverMultistepScheduler` (DPM-Solver++, second order).

## Files

| File                              | What it holds                                                                                                                                                                                                                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text.f32`                        | CLIP's last hidden state for the empty prompt and the prompt: first 8 of 768 numbers and the norm, at all 77 positions                                                                                                                                                                                              |
| `trajectory.f32`                  | the DDIM run at every step: z_t, ε(∅), ε(prompt), the guided ε̂, x̂₀, the sinusoid and the timestep embedding                                                                                                                                                                                                         |
| `samplers.f32`                    | the Euler and DPM-Solver++ runs: z_t and x̂₀ at every step                                                                                                                                                                                                                                                           |
| `step-TTT.f32`                    | DDIM steps t = 901, 501, 101: every block's RMS over channels for both passes; `down.1.resnet.0` stage by stage; `up.1.transformer.0` (8 × 8) stage by stage with all 8 heads' self- and cross-attention; every transformer block's cross-attention averaged over heads, per prompt token, plus the padding's total |
| `vae.f32`                         | the VAE on a photo: the latent distribution's mean and standard deviation, and mean × 0.18215                                                                                                                                                                                                                       |
| `*-final.png`                     | each run's image, as the pipeline returns it                                                                                                                                                                                                                                                                        |
| `*-x0.jpg`, `ddim-xt.jpg`         | 128 px JPEG decodes of every step's x̂₀ (and of DDIM's z_t), by the pipeline's VAE                                                                                                                                                                                                                                   |
| `vae-input.png`, `vae-output.png` | the photo as the VAE saw it, and its reconstruction                                                                                                                                                                                                                                                                 |

The photo is [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg) by Jon Sullivan,
public domain, from `static/architectures/sam-beagle.jpg`: the centre square, resized to 256 × 256
with Lanczos.

Attention probabilities are rebuilt as softmax(q·kᵀ/√d) from the captured q and k and checked
against the model's attention output. Each step is checked against its sampler's update applied to
the recorded noise prediction, the guidance mix against the two recorded predictions, and each
ResNet and transformer stage against a float64 rebuild from the stage before; `validation` in the
manifest lists every check's largest error.
