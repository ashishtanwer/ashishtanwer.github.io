# A DCGAN trained here on MNIST

`manifest.json`, `gan.f32` and `weights.f32` record a small DCGAN trained for this tab by
`scripts/before-transformers/record_gan.py` with the code of `pytorch/before_transformers_gan.py`. The generator turns 64
standard normal numbers into a 28 × 28 digit with three transposed convolutions (1 × 1 → 7 × 7 → 14 × 14 → 28 × 28,
128 and 64 channels, batch norm and ReLU between, tanh at the end; 533,889 parameters); the discriminator is three
convolutions with leaky ReLU (64 and 128 channels; 138,561 parameters). Both use Adam at 2e-4 with β₁ 0.5, batch 64,
15 epochs (14,055 alternating steps), seed 0, DCGAN's initialisation. Training ran on the Mac's GPU (MPS, about three
minutes); everything recorded was computed on the CPU in float32 from checkpoints saved during the run. The class
histograms come from the CNN tab's trained network.

## Data

The 60,000 MNIST training digits (Yann LeCun, Corinna Cortes and Christopher J. C. Burges,
<http://yann.lecun.com/exdb/mnist/>), licensed [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) by
its authors, read from the Hugging Face copy
[ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist/tree/77f3279092a1c1579b2250db8eafed0ad422088c) at revision
`77f3279092a1c1579b2250db8eafed0ad422088c` (whose card says MIT); pixels / 127.5 − 1, labels unused in training. The
recording holds the first 64 test digits and the two recorded batches of 64 training digits, as pixel values.

## Files

| Tensor                                                    | What it holds                                                                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `grid.z`, `grid.images`, `grid.logit`                     | 64 fixed noise vectors, their images at every checkpoint and the discriminator's logit on them                     |
| `real.x`, `real.logit`                                    | the first 64 test digits and the discriminator's logit on them at every checkpoint                                 |
| `scoreHist`                                               | at every checkpoint, D(x) in 20 bins for the 10,000 test digits and for 10,000 fresh samples                       |
| `layers.<u1 a1 u2 a2 u3 image>`                           | six grid vectors through the trained generator layer by layer (before and after batch norm + ReLU)                 |
| `disc.<x h1 h2 logit>`                                    | one real and one generated digit through the trained discriminator                                                 |
| `walk.<images logit probs zNorm>`                         | eleven points on a straight line between two noise vectors: images, discriminator logits, CNN probabilities, ‖z‖   |
| `update<n>.<real z fake fakeAfter …Logit…>`               | the updates at steps 200 and 7,000: the batch, the noise, the fakes before and after, every score before and after |
| `series.<lossD lossG realScore fakeScore fakeScoreAfter>` | every step's losses and mean scores; `gradD`, `gradG`: the norm of each network's gradient                         |
| `weights.f32`                                             | the final generator (with batch-norm running averages) and discriminator                                           |

The manifest also holds each checkpoint's class histogram of 10,000 samples by the CNN tab's network, its confidence and
the samples' mean pairwise distance, and `validation`: each recorded update against `nn.BCEWithLogitsLoss` +
`torch.optim.Adam` from copies, the drawer's transposed convolution against `F.conv_transpose2d`, and the CNN
classifier against the CNN tab's recorded logits.
