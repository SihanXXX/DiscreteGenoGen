# Generative Models for Synthetic Genotype Simulation

This repository contains code for four generative models used to generate synthetic genotype data: **VAE**, **GAN+GumbelSoftmax**, **WGAN+GumbelSoftmax**, and **PCA+Diffusion**. Each subrepository provides:
- **Model Architecture**: The code for building the respective generative model.
- **Model Configuration**: The best params configuration for dataset.
- **Model Training**: The training procedure for each model.
- **Pretrained Models**: Trained generative models ready to generate synthetic genotype data.

## Subrepositories

This repository is organized into four submodules, each corresponding to a different generative model:

### 1. [VAE(Variational Autoencoder)](./VAE)
### 2. [GAN_GS(GAN with Gumbel Softmax)](./GAN_GS)
### 3. [WGAN_GS(Wasserstein GAN with Gumbel Softmax)](./WGAN_GS)
### 4. [PCA_DM(PCA-based Latent Diffusion Model)](./PCA_DM)