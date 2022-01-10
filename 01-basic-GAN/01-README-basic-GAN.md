## Generative Adversarial Neural Networks
This project uses Generative Adversarial Neural Networks, or GANs. This is a fairly new type of network, proposed in 2014 by [Goodfellow et al](https://arxiv.org/abs/1406.2661). They proposed an adversarial architecture where two network components, the Generator and the Discriminator, are trained in parallel, competing in a zero sum game.

The Generator aims to produce new datapoints similar to the training dataset, while the Discriminator aims to distinguish the datapoints created by the Generator from the training points.

## Network Architectures

There are currently two GAN architectures, based on [this tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). One produces images of size 64 by 64 and the other produces images of size 256 by 256. Their architectures are detailed below.

The specific architecture used for the 64 by 64 GAN can be found [here](64x64Architecture.md) and the architecture of the 256 by 256 GAN can be found [here](256x256Architecture.md).
