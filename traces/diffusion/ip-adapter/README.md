# Recorded IP-Adapter on Stable Diffusion 1.5

`manifest.json`, its `.f32` files and images are recorded from
[h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter/tree/018e402774aeeddd60609b4ecdb7e298259dc729)
(Ye et al., [arXiv:2308.06721](https://arxiv.org/abs/2308.06721)), revision
`018e402774aeeddd60609b4ecdb7e298259dc729`, Apache-2.0: `models/ip-adapter_sd15.safetensors` and its
`models/image_encoder` (OpenCLIP ViT-H/14), on
[stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14)
(CreativeML OpenRAIL-M). The files hold activations and generated images, never the weights.
`scripts/diffusion/record_ip_adapter.py` writes them through diffusers' own `StableDiffusionPipeline` with
`load_ip_adapter` (safety checker not loaded); hooks only observe, and unhooked calls give identical images.

## Runs

The whole photo below as the image prompt (CLIP's processor takes its centre 224 × 224), one text prompt
(`prompt` in the manifest), the empty prompt and an all-zero image embedding for classifier-free guidance at
scale 7.5, 384 × 384 pixels, 20 DDIM steps (the SD 1.5 tab's sampler) and
`torch.Generator("cpu").manual_seed(0)`. Image-prompt scale 0.5 is recorded in depth; the same seed then runs
at scales 0, 0.25, 0.5, 0.75 and 1, and once more with SD 1.5 alone before the adapter is loaded.

## Files

| File                 | What it holds                                                                                                                                                                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image.f32`          | the image encoder's last-layer attention from the class token (mean over heads), the 1024-wide image embedding, and the 4 image tokens for the zero embedding and the photo (first 8 of 768 numbers, norms)                                             |
| `text.f32`           | CLIP's last hidden state for the empty prompt and the prompt: first 8 of 768 numbers and the norm, at all 77 positions                                                                                                                                  |
| `trajectory.f32`     | at every step: z_t, ε(∅ + zero image), ε(prompt + photo) and the guided ε̂                                                                                                                                                                               |
| `step-TTT.f32`       | steps t = 901, 501, 101, prompt pass: every block's RMS over channels; every cross-attention's attention to the 4 image tokens (mean over heads); in `up.1.transformer.0`, all 8 heads' text and image attention and each branch's output as an RMS map |
| `photo.jpg`          | the image prompt                                                                                                                                                                                                                                        |
| `clip-input.png`     | the 224 × 224 crop CLIP's processor makes of it                                                                                                                                                                                                         |
| `scale-S.jpg`        | the final image at image-prompt scale S (0 is SD 1.5's own image)                                                                                                                                                                                       |
| `x0-S.jpg`, `xt.jpg` | 128 px JPEG decodes of every step's x̂₀ at scales 1, 0.5 and 0, and of z_t, by the pipeline's VAE                                                                                                                                                        |

Each manifest entry `details[].layers` gives, for all 16 cross-attentions, the RMS of the text branch and of
the scaled image branch.

The photo is [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg) by Jon Sullivan, public
domain, from `static/architectures/sam-beagle.jpg`.

Each DDIM step is checked against the update applied to the recorded ε̂, the guidance against the two
recorded predictions, the image embedding and the 4 tokens against float64 rebuilds from the encoder's and
projection's weights, the encoder's last attention against its output, and every decoupled cross-attention
against softmax(q·k_textᵀ/√d)·v_text + λ·softmax(q·k_imageᵀ/√d)·v_image rebuilt from the captures;
`validation` in the manifest lists every check's largest error.
