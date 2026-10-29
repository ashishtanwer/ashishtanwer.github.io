# Recorded DDPM on CIFAR-10

`manifest.json` and its `.f32` files are recorded from the pretrained
[google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32/tree/267b167dc01f0e4e61923ea244e8b988f84deb80)
checkpoint (Ho, Jain & Abbeel, [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)), revision
`267b167dc01f0e4e61923ea244e8b988f84deb80`, Apache-2.0. The files hold activations, never the
weights. `scripts/diffusion/record_ddpm.py` writes them through diffusers' own `UNet2DModel`,
`DDPMScheduler` and `DDPMPipeline`; hooks only observe (unhooked runs give the same ε̂ and the same
images), and the manifest records the package versions, the script and helper hashes and the weights'
SHA-256.

## Inputs

- Four CIFAR-10 test images, numbers 1, 6, 7 and 13 (ship, automobile, frog, horse), read from
  [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10/tree/0b2714987fa478483af9968de7c934580d0bb9a2)
  (Krizhevsky, _Learning Multiple Layers of Features from Tiny Images_, 2009; the dataset card lists
  the license as unknown). The manifest keeps each PNG's SHA-256.
- One noise ε per image from `torch.Generator().manual_seed(1)`.
- The reverse run: `DDPMPipeline(batch_size=16, generator=torch.Generator("cpu").manual_seed(0))`,
  all 1000 steps.

## Files

| File           | What it holds                                                                                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `schedule.f32` | β_t and ᾱ_t for t = 0 … 999, as the scheduler holds them                                                                                                                                                     |
| `forward.f32`  | x₀, ε, and at t = 0, 50, 100, 250, 500, 750, 999: x_t (`DDPMScheduler.add_noise`), the U-Net's ε̂ and the scheduler's clipped x̂₀; the sinusoid and the timestep embedding; the ResNet block's time projection |
| `image-N.f32`  | the U-Net on image N at those seven t: every block's RMS over channels at each pixel; `down.1.resnet.0` stage by stage (4 channels and the RMS of all); 16 query pixels' self-attention rows at 16 × 16      |
| `reverse.f32`  | the 16 final images, and for sample 8 at 28 of the 1000 steps: x*t, ε̂, x̂₀, the drawn noise z and x*{t−1}                                                                                                     |

Attention probabilities are not formed by the model's fused attention; the recorder rebuilds
softmax(q·kᵀ/√256) from the captured q and k and checks that, times v, it gives the model's
attention output. The noise z of each reverse step is redrawn from the generator state the
scheduler used, and every one of the 1000 steps is checked against
x\_{t−1} = μ(x_t, x̂₀) + √β_t · z. `validation` in the manifest lists every check's largest error.
