## Generative Adversarial Neural Networks
This project uses Generative Adversarial Neural Networks, or GANs. This is a fairly new type of network, proposed in 2014 by [Goodfellow et al](https://arxiv.org/abs/1406.2661). They proposed an adversarial architecture where two network components, the Generator and the Discriminator, are trained in parallel, competing in a zero sum game.

The Generator aims to produce new datapoints similar to the training dataset, while the Discriminator aims to distinguish the datapoints created by the Generator from the training points.

## Network Architectures

There are currently two GAN architectures, based on [this tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). One produces images of size 64 by 64 and the other produces images of size 256 by 256. Their architectures are detailed below:

### 64x64

Generator:
| Layer                            |  Output Size  |
|----------------------------------|:-------------:|
| Input                            |  100 x 1 x 1  |
| TransConv, BatchNorm, Leaky Relu |  512 x 4 x 4  |
| TransConv, BatchNorm, Leaky Relu |  256 x 8 x 8  |
| TransConv, BatchNorm, Leaky Relu | 128 x 16 x 16 |
| TransConv, BatchNorm, Leaky Relu |  64 x 32 x 32 |
| TransConv, Leaky Relu            |  1 x 64 x 64  |

Discriminator:
| Layer                       |  Output Size  |
|-----------------------------|:-------------:|
| Input                       |  1 x 64 x 64  |
| Conv, Leaky Relu            |  64 x 32 x 32 |
| Conv, BatchNorm, Leaky Relu | 128 x 16 x 16 |
| Conv, BatchNorm, Leaky Relu |  256 x 8 x 8  |
| Conv, BatchNorm, Leaky Relu |  512 x 4 x 4  |
| Conv, Sigmoid               |       1       |

### 256x256

**Generator:**
| Layer                            |   Output Size  |
|----------------------------------|:--------------:|
| Input                            |   100 x 1 x 1  |
| TransConv, BatchNorm, Leaky Relu |   512 x 4 x 4  |
| TransConv, BatchNorm, Leaky Relu |   256 x 8 x 8  |
| TransConv, BatchNorm, Leaky Relu |  128 x 16 x 16 |
| TransConv, BatchNorm, Leaky Relu |  64 x 32 x 32  |
| TransConv, BatchNorm, Leaky Relu |  32 x 64 x 64  |
| TransConv, BatchNorm, Leaky Relu | 16 x 128 x 128 |
| TransConv, BatchNorm, Leaky Relu |  8 x 256 x 256 |
| Conv, Tanh                       |  1 x 256 x 256 |

Discriminator:
| Layer                                 |  Output Size  |
|---------------------------------------|:-------------:|
| Input                                 | 1 x 256 x 256 |
| Conv, Leaky Relu, Max Pool            | 8 x 127 x 127 |
| Conv, BatchNorm, Leaky Relu, Max Pool |  16 x 63 x 63 |
| Conv, BatchNorm, Leaky Relu, Max Pool |  32 x 30 x 30 |
| Conv, BatchNorm, Leaky Relu, Max Pool |  64 x 14 x 14 |
| Conv, BatchNorm, Leaky Relu, Max Pool |  128 x 6 x 6  |
| Conv, BatchNorm, Leaky Relu, Max Pool |   256 x 2 x2  |
| Flatten, Linear Layer, Leaky Relu     |      128      |
| Linear Layer, Sigmoid Function        |       1       |
