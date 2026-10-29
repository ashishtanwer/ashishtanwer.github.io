# Recorded ControlNet (canny) on Stable Diffusion 1.5

`manifest.json`, its `.f32` files and images are recorded from the pretrained
[lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny/tree/7f2f69197050967007f6bbd23ab5e52f0384162a)
ControlNet (Zhang et al., [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)), revision
`7f2f69197050967007f6bbd23ab5e52f0384162a`, on
[stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14),
revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, both under the
[CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license).
The files hold activations and generated images, never the weights.
`scripts/diffusion/record_controlnet.py` writes them through diffusers' own
`StableDiffusionControlNetPipeline` (safety checker not loaded); hooks only observe, and an
unhooked call at scale 1 gives the identical image.

## Runs

The prompt `a tiger cub in a garden, photo`, the empty prompt for classifier-free guidance at
scale 7.5, 384 × 384 pixels (a 4 × 48 × 48 latent), `DDIMScheduler` with 20 steps and
`torch.Generator("cpu").manual_seed(0)`, at conditioning scales 0, 0.25, 0.5, 0.75, 1 and 1.5;
and once through `StableDiffusionPipeline` without the ControlNet, which gives exactly the scale-0
image. The control image is `cv2.Canny(photo, 100, 200)`, as on the model card, repeated as RGB.

## Files

| File                   | What it holds                                                                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `edges.f32`            | the edge encoder's output (320 × 48 × 48): its RMS over channels and the four channels that vary most; the copy's `conv_in(zₜ)` at the three detail steps                                                       |
| `trajectory.f32`       | the scale-1 run at every step: zₜ, ε(∅), ε(prompt) and the guided ε̂; the RMS of every residual reaching the U-Net at scales 1 and 0.5                                                                           |
| `step-TTT.f32`         | t = 901, 501, 101: every U-Net block's RMS map (prompt pass); for each of the 12 skips and the middle, the copy's feature, the residual, the skip with the residual added, as RMS maps, and ‖residual‖ / ‖skip‖ |
| `scale-*.png`          | the final image at each conditioning scale, as the pipeline returns it                                                                                                                                          |
| `x0-*.jpg`, `xt.jpg`   | 128 px JPEG decodes of every step's x̂₀ at scales 1, 0.5 and 0, and of zₜ at scale 1, by the pipeline's VAE                                                                                                      |
| `photo.png`, `canny-*` | the photo as the detector saw it, the gradient \|dx\| + \|dy\| (scaled to its maximum), the thinned maxima above 100, hysteresis (white > 200, amber kept, red dropped) and the edge map                        |

The photo is [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg) by Jon Sullivan,
public domain, from `static/architectures/sam-beagle.jpg`: the centre square, resized with Lanczos.

The canny stages come from the tab's PyTorch code (`pytorch/diffusion_controlnet.py`) and its edge
map equals OpenCV's. Each recorded stage is checked against a float64 rebuild from the stage before:
the edge encoder, the copy's input, every zero convolution, the residuals the U-Net receives, the skips
and middle the up path reads, the guidance mix and the DDIM step. At step 1 every scale's residuals
are exactly that multiple of scale 1's. `validation` in the manifest lists every check's largest error.
