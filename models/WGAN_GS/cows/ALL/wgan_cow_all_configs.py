import torch

# WGAN config 
CONFIG_WGAN_COW_ALL= dict()

# Data
CONFIG_WGAN_COW_ALL['x_dim'] = 50161

# Architecture
CONFIG_WGAN_COW_ALL['latent_dim'] = 256
CONFIG_WGAN_COW_ALL['pheno_dim'] = 1
CONFIG_WGAN_COW_ALL["proportional_neuron"] = True # If sets to True then the neuron size in each layer is proportional to the snp length
CONFIG_WGAN_COW_ALL['hidden_dim1_g'] = 512
CONFIG_WGAN_COW_ALL['hidden_dim2_g'] = 1024
CONFIG_WGAN_COW_ALL['hidden_dim3_g'] = 4096
CONFIG_WGAN_COW_ALL['hidden_dim1_d'] = 4096
CONFIG_WGAN_COW_ALL['hidden_dim2_d'] = 1024
CONFIG_WGAN_COW_ALL['hidden_dim3_d'] = 512

if CONFIG_WGAN_COW_ALL["proportional_neuron"]:
    CONFIG_WGAN_COW_ALL['hidden_dim1_g'] = (3 * CONFIG_WGAN_COW_ALL['x_dim']) // 144
    CONFIG_WGAN_COW_ALL['hidden_dim2_g'] = (3 * CONFIG_WGAN_COW_ALL['x_dim']) // 72
    CONFIG_WGAN_COW_ALL['hidden_dim3_g'] = (3 * CONFIG_WGAN_COW_ALL['x_dim']) // 36
    CONFIG_WGAN_COW_ALL["hidden_dim1_d"] = (3 * CONFIG_WGAN_COW_ALL['x_dim']) // 36
    CONFIG_WGAN_COW_ALL["hidden_dim2_d"] = (3 * CONFIG_WGAN_COW_ALL['x_dim']) // 72
    CONFIG_WGAN_COW_ALL["hidden_dim3_d"] = (3 * CONFIG_WGAN_COW_ALL['x_dim']) // 144

# Training
CONFIG_WGAN_COW_ALL['activation'] = 'leaky_relu'
CONFIG_WGAN_COW_ALL['negative_slope'] = 0.05
CONFIG_WGAN_COW_ALL['optimizer'] = 'adam'
CONFIG_WGAN_COW_ALL['lr_g'] = 1e-4
CONFIG_WGAN_COW_ALL['lr_d'] = 1e-4
CONFIG_WGAN_COW_ALL['batch_size'] = 128
CONFIG_WGAN_COW_ALL['epochs'] = 1000
CONFIG_WGAN_COW_ALL['n_critic'] = 5
CONFIG_WGAN_COW_ALL['lambda_penalty'] = 10
CONFIG_WGAN_COW_ALL['step'] = 10
CONFIG_WGAN_COW_ALL['init_temp'] = 1
CONFIG_WGAN_COW_ALL['final_temp'] = 0.1

# Device
device = torch.device("cuda:0")