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


Each subdirectory contains:

- `{model_type}_model.py`  
  Defines the architecture of the corresponding generative model.

- `{model_type}_{dataset}_configs.py`  
  Contains the best hyperparameter configuration identified during our experiments.

- `train.py`  
  Script used to train the model.

Sometimes a pretrained model (`.pth` file) is also included. These models can be loaded directly in the [Demo Notebooks](../demo) to simulate synthetic genotypes.

## Model Training and Selection
We strongly recommend monitoring the training of generative models using [Weights & Biases](https://wandb.ai/site/). For model selection, we suggest choosing the checkpoint where the F1 score stops improving.