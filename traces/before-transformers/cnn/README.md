# A convolutional network trained here on MNIST

`manifest.json` and `cnn.f32` record a LeNet-style convolutional network (6 kernels 5 × 5 with padding 2, ReLU, 2 × 2
max pooling, 16 kernels 5 × 5 × 6, ReLU, pooling, one dense layer 400 → 10; 6,582 parameters) trained for this tab on
the Mac's CPU in float32 by `scripts/before-transformers/record_cnn.py`, with the code of
`pytorch/before_transformers_cnn.py`: autograd's gradients and plain SGD (lr 0.1, no momentum, no weight decay),
batches of 64 in the MLP tab's order, 3 epochs (2,811 steps), seed 0, PyTorch's default initial weights. It takes about
a minute and repeats bit for bit. Test accuracy goes from 12.3 % to 97.9 % (the MLP tab: 95.4 % with 52,650 parameters).

## Data

The MNIST database of handwritten digits (Yann LeCun, Corinna Cortes and Christopher J. C. Burges,
<http://yann.lecun.com/exdb/mnist/>), licensed
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) by its authors, read from the Hugging Face copy
[ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist/tree/77f3279092a1c1579b2250db8eafed0ad422088c) at revision
`77f3279092a1c1579b2250db8eafed0ad422088c` (whose card says MIT). Pixels are divided by 255; nothing else is done to
them. The recording holds 10 test digits (the first of each class, the MLP tab's ten) and the two recorded batches of
64 training digits, as pixel values; the manifest lists their indices.

## Files

| Tensor                                              | What it holds                                                                                                                        |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `ckpt.<param>`                                      | every kernel, bias and dense weight after steps 0, 1, 10, 100, 500, 1000 and 2811                                                    |
| `digit.x`, `digit.<z1 p1 win1 z2 p2 win2 logits p>` | the ten test digits, and at each checkpoint both convolutions' maps before ReLU, the pooled maps, each pooling window's winning cell |
| `digit.loss`                                        | each digit's loss at each checkpoint                                                                                                 |
| `erf`                                               | at the end, per digit: the gradient of each layer's strongest unit (and of the label's logit) with respect to the 28 × 28 pixels     |
| `update<n>.x`, `.grad.<param>`, `.after.<param>`    | the first update and the one at step 1000: the batch, the gradient of every parameter, the weights after                             |
| `update<n>.perImage`, `update<n>.gradZ1`            | each image's share of the first layer's kernel gradients, and ∂L/∂(first-layer maps) of the batch's first 16 images                  |
| `series.loss`, `series.accuracy`, `series.gradNorm` | every step: the minibatch's loss and accuracy and ‖∂L/∂W‖ of conv 1, conv 2 and the dense layer                                      |

The manifest also holds the test and training-subset loss and accuracy every 25 steps, the MLP tab's numbers (from its
manifest, pinned by SHA-256), both convolutions as dense matrices (entries, non-zero entries, distinct values), the
receptive-field arithmetic and `validation`: the drawer's convolution against `F.conv2d`, its pooling against
`F.max_pool2d`, its kernel gradient against autograd, the SGD step against `torch.optim.SGD` and against w − lr · g, the
forward pass against float64 and each receptive-field gradient against its window.
