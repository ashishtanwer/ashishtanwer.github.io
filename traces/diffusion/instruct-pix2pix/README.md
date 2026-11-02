# Recorded InstructPix2Pix

`manifest.json`, its `.f32` files and images are recorded from the pretrained
[timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix/tree/31519b5cb02a7fd89b906d88731cd4d6a7bbf88d)
checkpoint (Brooks, Holynski and Efros, [arXiv:2211.09800](https://arxiv.org/abs/2211.09800)), revision
`31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`, under the MIT license given in its model card. The model is
fine-tuned from Stable Diffusion 1.5 (CreativeML OpenRAIL-M). The files hold activations and generated
images, never the weights. `scripts/diffusion/record_instruct_pix2pix.py` writes them through diffusers' own
`StableDiffusionInstructPix2PixPipeline` (safety checker not loaded); hooks only observe, and an unhooked
call gives the identical image.

## Runs

The photo below at 256 × 256 (the resolution the model was trained at), one instruction (`instruction` in the
manifest), 20 steps of the model's own `EulerAncestralDiscreteScheduler` and
`torch.Generator("cpu").manual_seed(0)`. Text guidance 7.5 and image guidance 1.5 are recorded in depth; the
same seed then runs at every pair of text scales 3, 5, 7.5, 10 and image scales 1, 1.25, 1.5, 2.

## Files

| File                   | What it holds                                                                                                                                                                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `text.f32`             | CLIP's last hidden state for the empty prompt and the instruction: first 8 of 768 numbers and the norm, at all 77 positions                                                                                                                                                |
| `image.f32`            | the photo's latent as the U-Net reads it (the VAE distribution's mode, not × 0.18215), and ‖W‖ of conv_in over each of its 8 input channels                                                                                                                                |
| `trajectory.f32`       | at every step: z_t, the three noise predictions (instruction + photo, ∅ + photo, ∅ with an all-zero image latent), the guided ε̂, and the noise Euler ancestral adds                                                                                                        |
| `step-NN.f32`          | steps 2, 10 and 18, instruction + photo pass: every block's RMS over channels; conv_in's output split into the noisy latent's and the photo latent's parts; every transformer block's cross-attention averaged over heads, per instruction token, plus the padding's total |
| `photo.png`            | the photo as the pipeline saw it                                                                                                                                                                                                                                           |
| `edit.png`             | the edit at the default scales, as the pipeline returns it                                                                                                                                                                                                                 |
| `sweep-T-I.jpg`        | the edit at text scale T and image scale I                                                                                                                                                                                                                                 |
| `x0-T-I.jpg`, `xt.jpg` | 128 px JPEG decodes of every step's x̂₀ for three pairs of scales, and of z_t, by the pipeline's VAE                                                                                                                                                                        |

The photo is [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg) by Jon Sullivan, public
domain, from `static/architectures/sam-beagle.jpg`: the centre square, resized with Lanczos.

Each step is checked against Euler ancestral's update applied to the recorded ε̂ and the recorded noise, the
noise against the seeded generator, the guidance against the three recorded predictions, conv_in's parts
against its output, and every cross-attention against softmax(q·kᵀ/√d) rebuilt from the captured q and k;
`validation` in the manifest lists every check's largest error.
