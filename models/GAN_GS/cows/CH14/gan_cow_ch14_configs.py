import torch

# WGAN config 
CONFIG_GAN_COW_CH14= dict()

# Data
CONFIG_GAN_COW_CH14['x_dim'] = 1771
CONFIG_GAN_COW_CH14['snp_repr'] = "one-hot"

# Architecture
CONFIG_GAN_COW_CH14['latent_dim'] = 128
CONFIG_GAN_COW_CH14['pheno_dim'] = 1
CONFIG_GAN_COW_CH14["proportional_neuron"] = True # If sets to True then the neuron size in each layer is proportional to the snp length
CONFIG_GAN_COW_CH14['hidden_dim1_g'] = 512
CONFIG_GAN_COW_CH14['hidden_dim2_g'] = 1024
CONFIG_GAN_COW_CH14['hidden_dim1_d'] = 1024
CONFIG_GAN_COW_CH14['hidden_dim2_d'] = 512

if CONFIG_GAN_COW_CH14["proportional_neuron"]:
    CONFIG_GAN_COW_CH14['hidden_dim1_g'] = (3 * CONFIG_GAN_COW_CH14['x_dim']) // 6
    CONFIG_GAN_COW_CH14['hidden_dim2_g'] = (3 * CONFIG_GAN_COW_CH14['x_dim']) // 3
    CONFIG_GAN_COW_CH14["hidden_dim1_d"] = (3 * CONFIG_GAN_COW_CH14['x_dim']) // 5
    CONFIG_GAN_COW_CH14["hidden_dim2_d"] = (3 * CONFIG_GAN_COW_CH14['x_dim']) // 10

# Training
CONFIG_GAN_COW_CH14['activation'] = 'leaky_relu'
CONFIG_GAN_COW_CH14['negative_slope'] = 0.05
CONFIG_GAN_COW_CH14['optimizer'] = 'adam'
CONFIG_GAN_COW_CH14['lr_g'] = 5e-4
CONFIG_GAN_COW_CH14['lr_d'] = 1e-4
CONFIG_GAN_COW_CH14['batch_size'] = 256
CONFIG_GAN_COW_CH14['epochs'] = 1000
CONFIG_GAN_COW_CH14['step'] = 10
CONFIG_GAN_COW_CH14['init_temp'] = 1
CONFIG_GAN_COW_CH14['final_temp'] = 0.1

# Device Configuration
device = torch.device("cuda:1")