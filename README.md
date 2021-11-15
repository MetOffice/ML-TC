

# ML-TC



<h4 align="center">
🤖+🌀 Machine Learning explorations with tropical cyclone data
</h4>

<p align="center">
  <a href="https://github.com/MetOffice/ML-TC/LICENSE">
      <img src="https://img.shields.io/github/license/MetOffice/ML-TC.svg?style=flat-square"
          alt="GitHub license" /></a>

  <a href="https://GitHub.com/MetOffice/ML-TC/graphs/contributors/">
       <img src="https://img.shields.io/github/contributors/MetOffice/ML-TC.svg?style=flat-square"
            alt="GitHub contributors" /></a>
  <a href="">
      <img src="https://img.shields.io/github/last-commit/MetOffice/ML-TC?style=flat-square"
          alt="GitHub last-commit" /></a>
</p>

This repository contains code for exploring tropical cyclone data using machine learning techniques.  The machine learning frameworks in this repo are prinicpally based in python, in particular **[pytorch](https://pytorch.org/)**.  Training data comes from a variety of [Met Office](https://www.metoffice.gov.uk/) high resolution models based on the [Unified Model (UM)](https://www.metoffice.gov.uk/research/approach/modelling-systems/unified-model/index).

## Contents
Key file and folders

|   |Description |
|--:|:---|
| [example_notebook.ipynb](dir/example_notebook.ipynb)  |Description of file/directory   |

## Contributing
Information on how to contribute can be found in the [Contributing guide](CONTRIBUTING.md).

## Getting Started

Set-up a coda environment for this repository based on the `requirements.yml` file:

```bash
conda env create --file requirements.yml
```

This environment requires ~10GB of disk space.  Then activate the conda environment:

```bash
conda activate ml-tc
```

## Data

Current sources of data used by this repo include:

* [Bangladesh Topical Cyclone Data](https://doi.org/10.5281/zenodo.3600201.) via Zenodo. This data is distributed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/), please cite https://doi.org/10.1038/s41597-021-00847-5 if used.

<h5 align="center">
<img src="etc/MO_MASTER_black_mono_for_light_backg_RBG.png" width="200" alt="Met Office"> <br>
&copy; British Crown Copyright, Met Office
</h5>

## Generative Adversarial Neural Networks
This project uses Generative Adversarial Neural Networks, or GANs. This is a fairly new type of network, proposed in 2014 by [Goodfellow et al](https://arxiv.org/abs/1406.2661). They proposed an adversarial architecture where two network components, the Generator and the Discriminator, are trained in parallel, competing in a zero sum game.

The Generator aims to produce new datapoints similar to the training dataset, while the Discriminator aims to distinguish the datapoints created by the Generator from the training points.

## Network Architectures

There are currently two GAN architectures, based on [this tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). One produces images of size 64 by 64 and the other produces images of size 256 by 256. Their architectures are detailed below.

The specific architecture used for the 64 by 64 GAN can be found [here](64x64Architecture.md)
### 256 by 256 GAN
Generator:
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
