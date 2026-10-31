# Recorded ResNet-50 on one photo

`manifest.json`, `resnet.f32` and `photo.png` are recorded from the pretrained
[microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50/tree/34c2154c194f829b11125337b98c8f5f9965ff19)
(He, Zhang, Ren and Sun, [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)), revision
`34c2154c194f829b11125337b98c8f5f9965ff19`, under the Apache-2.0 license. The files hold activations, gradients and the
64 stem kernels, never the rest of the weights. `scripts/before-transformers/record_resnet.py` runs the photo through
the drawer's own ResNet-50 (`pytorch/before_transformers_resnet.py`) with the released weights, in float32 on the CPU,
and checks it against transformers' `ResNetForImageClassification` (logits and stage outputs equal, max |Δ| 0).

## The photo

`static/architectures/sam-beagle.jpg`, [Dog beagle](https://commons.wikimedia.org/wiki/File:Dog_beagle.jpg) by Jon
Sullivan, public domain (Wikimedia Commons; SHA-256 of the original download in the manifest). `photo.png` is the
224 × 224 view the network sees: shortest side to 256 (bicubic), the centre 224 × 224, as the model's
`ConvNextImageProcessor` does, before the ImageNet mean/std normalisation. Top class: beagle, p 0.992.

## Files

| Tensor                                        | What it holds                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `stem.kernels`                                | the 64 learned 7 × 7 × 3 kernels of the stem convolution                                                                  |
| `stem.conv`, `stem.pool`, `…Rms`              | six channels of the stem convolution's output (112 × 112) and of the max pool's (56 × 56), and the RMS over all 64        |
| `stage<s>.out`, `stage<s>.rms`                | eight channels of each stage's output and the RMS over all its channels                                                   |
| `block<k>.<x reduce conv f shortcut sum out>` | every one of the 16 bottleneck blocks stage by stage, four channels each (`f`, `shortcut`, `sum`, `out` share channels)   |
| `block<k>.<stage>Rms`                         | the RMS over all channels of each of those stages                                                                         |
| `erf<i>`                                      | \|∂(all channels of the centre pixel) / ∂ input\| summed over RGB, for the stem convolution, the pool and the four stages |
| `pooled`, `logits`, `votes`, `stage4.voters`  | the 2,048 pooled features, the 1,000 logits, w_c · pooled for the top class, and the 7 × 7 maps of its 8 largest voters   |

Channels are the ones whose values vary most over the pixels of this photo; the manifest lists them. Per block it also
holds ‖F(x)‖, ‖shortcut(x)‖, the share ReLU zeroes, and the gradient of the top class's logit: arriving at the block's
output, sent back through the skip, through F, and the cosine between those two parts. `receptiveField` is the drawer's
formula for every layer, `erf` the window each recorded gradient stays inside. `validation` holds every check: the
preprocessing against the processor, the model against transformers, every block recomputed in float64 with `F.conv2d`
and the batch-norm formula, the two gradient parts against the full gradient, and the logit against bias + Σ votes.
