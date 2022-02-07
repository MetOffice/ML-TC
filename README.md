

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

<h4 align="center">
<a href=""><img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/></a> <a href=""><img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/></a>
</h4>

This repository contains code for exploring tropical cyclone data using machine learning techniques.  The machine learning frameworks in this repo are prinicpally based in python, in particular **[pytorch](https://pytorch.org/)**.  Training data comes from a variety of [Met Office](https://www.metoffice.gov.uk/) high resolution models based on the [Unified Model (UM)](https://www.metoffice.gov.uk/research/approach/modelling-systems/unified-model/index).

## Contents
Key file and folders

| Directory |Description |
|--:|:---|
| [01-basic-GAN](01-basic-GAN/01-README-basic-GAN.md) | Basic GAN with 64x64 and 256x256 architectures.   |

## Contributing
Information on how to contribute can be found in the [Contributing guide](CONTRIBUTING.md).

## Getting Started
Start by cloning this github repo:

```bash
git clone https://github.com/MetOffice/ML-TC.git
cd ML-TC
```

Then, set-up a conda environment for this repository.  Your conda environment will depend on if you have access to GPUs and CUDA.

### CPU architecture
For a **CPU only** conda environment, use the `requirements.yml` file:

```bash
conda env create --file requirements.yml
```

### GPU architecture
For a **GPU** optimised environment, check that you have access to GPUs and CUDA:

```bash
nvcc --version
lspci | grep -i nvidia
```

If these return scucesfully, then use the `requirements-gpu.yml` file:

```bash
conda env create --file requirements-gpu.yml
```

These environment requires ~10GB of disk space.  Then activate the conda environment:

```bash
conda activate ml-tc
```
```bash
conda activate ml-tc-gpu
```

To check if your GPU driver and CUDA is enabled and accessible by PyTorch, in `python` run:

```python
>>> import torch
>>> torch.cuda.is_available()
True
```

### Isambard :rocket:
Specific instructions for running on Isambard are detailed [here](isambard-getting-started.md)

## Data :floppy_disk:

Current sources of data used by this repo include:

* [Bangladesh Topical Cyclone Data](https://doi.org/10.5281/zenodo.3600201.) via Zenodo. This data is distributed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/), please cite https://doi.org/10.1038/s41597-021-00847-5 if used.

To download the data use Curl:
curl -# -o [cyclone].tar https://zenodo.org/api/files/476d7685-0a99-4f35-a32d-f6951e89dfec/tsens.[cyclone].tar.gz

where [cyclone] should be replaced with the capitalised name of the cyclone you want to download. 

The file getbengaldata.sh will download the cyclone data, unzip it, and run the data preparation script on it for the requested format and variables. The raw data will be temporarily stored in a folder called RawData while this runs, and the produced data will be stored in a folder called Data.

Example:

> ./getbengaldata.sh A "fg hur"

This will produce data in format A for the variables fg and hur.

<h5 align="center">
<img src="etc/MO_MASTER_black_mono_for_light_backg_RBG.png" width="200" alt="Met Office"> <br>
&copy; British Crown Copyright, Met Office
</h5>
