# Latent Diffusion Model with U-Net Architecture

## Overview

This repository contains code to train a Latent Diffusion Model using a U-Net neural network. The model consists of four decoders, a bottleneck layer, and four encoders. The training is performed on the CIFAR dataset.

## Dependencies

Make sure you have the following dependencies installed:

- [torch](https://pytorch.org/)
- [torchvision](https://pypi.org/project/torchvision/)
- [torchsummary](https://pypi.org/project/torchsummary/)

You can install them using:

pip install torch torchvision torchsummary

## Dataset

### CIFAR Dataset

The code is set up to use the CIFAR dataset. Download and preprocess the CIFAR dataset before running the code.

## Model Architecture

### U-Net Architecture

The Latent Diffusion Model uses a U-Net architecture with the following components:

- **Four Decoders:** These are the layers that upsample the features.
- **Bottleneck Layer:** This layer reduces the dimensionality and serves as the bottleneck of the U-Net.
- **Four Encoders:** These are the layers that downsample the features.

The architecture is designed to capture hierarchical features in the data and facilitate latent diffusion.

## Training

To train the model, follow these steps:

1. **Clone the repository:**

    git clone https://github.com/yourusername/latent-diffusion-model.git
    cd latent-diffusion-model

2. **Download and preprocess the CIFAR dataset.**

3. **Run the training script:**

    python train.py

    Make sure to adjust hyperparameters and paths in the script according to your requirements.

## Results

After training, you can evaluate the model and visualize the results. Check the `eval.ipynb` notebook for an example evaluation.

## Acknowledgments

This code is based on the Latent Diffusion Models and U-Net architecture. Special thanks to the contributors of [torch](https://pytorch.org/) and [torchvision](https://pypi.org/project/torchvision/) for providing essential tools for deep learning.

Feel free to reach out if you have any questions or issues! Contact: [amirzahedi0@gmail.com](mailto:amirzahedi0@gmail.com)
