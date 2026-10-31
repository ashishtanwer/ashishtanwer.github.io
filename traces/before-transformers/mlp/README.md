# A multilayer perceptron trained here on MNIST

`manifest.json` and `mlp.f32` record a 784 → 64 → 32 → 10 multilayer perceptron (ReLU, softmax and cross-entropy)
trained for this tab on the Mac's CPU in float32 by `scripts/before-transformers/record_mlp.py`, with the code of
`pytorch/before_transformers_mlp.py`: a hand-written backward pass and plain SGD (lr 0.1, no momentum, no weight
decay), batches of 64, 3 epochs (2,811 steps), seed 0, PyTorch's default initial weights. It takes seconds and repeats
bit for bit. Test accuracy goes from 8.9 % to 95.4 %.

## Data

The MNIST database of handwritten digits (Yann LeCun, Corinna Cortes and Christopher J. C. Burges,
<http://yann.lecun.com/exdb/mnist/>), licensed
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) by its authors, read from the Hugging Face copy
[ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist/tree/77f3279092a1c1579b2250db8eafed0ad422088c) at revision
`77f3279092a1c1579b2250db8eafed0ad422088c` (whose card says MIT). Pixels are divided by 255 and laid end to end; nothing
else is done to them. The recording holds 10 test digits (the first of each class), the 5 test images that drive each
first-layer unit hardest, and the two recorded batches of 64 training digits, as pixel values; the manifest lists their
indices.

## Files

| Tensor                                              | What it holds                                                                                              |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `ckpt.<param>`                                      | every weight and bias after steps 0, 1, 10, 100, 500, 1000 and 2811                                        |
| `digit.x`, `digit.<z1 h1 z2 h2 logits p>`           | the ten test digits, and at each checkpoint their activations layer by layer and the softmax               |
| `digit.grad.<logits h2 z2 h1 z1>`, `digit.loss`     | at each checkpoint, each digit's own backward pass (∂L/∂ every activation) and its loss                    |
| `update<n>.x`, `.grad.<param>`, `.after.<param>`    | the first update and the one at step 1000: the batch, the gradient of every parameter, the weights after   |
| `unit.classMean`, `unit.activeShare`                | at each checkpoint, each first-layer unit's mean activation per digit class and how often it is above zero |
| `unit.topX`, `unit.topValue`                        | at the end, the five test images that drive each first-layer unit hardest, and its activation on them      |
| `series.loss`, `series.accuracy`, `series.gradNorm` | every step: the minibatch's loss and accuracy and ‖∂L/∂W‖ of the three layers                              |

The manifest also holds the test and training-subset loss and accuracy every 25 steps, and `validation`: the
hand-written backward pass against autograd, the SGD step against `torch.optim.SGD` and against w − lr · g, the loss
against `F.cross_entropy` and the forward pass against float64.
