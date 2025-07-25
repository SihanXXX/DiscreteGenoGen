import torch

# WGAN config 
CONFIG_WGAN_HUMAN_EN = dict()

# Data
CONFIG_WGAN_HUMAN_EN['x_dim'] = 3493
CONFIG_WGAN_HUMAN_EN['snp_repr'] = "one-hot"
CONFIG_WGAN_HUMAN_EN['rescale_height'] = True

# Architecture
CONFIG_WGAN_HUMAN_EN['latent_dim'] = 128
CONFIG_WGAN_HUMAN_EN['sex_dim'] = 1
CONFIG_WGAN_HUMAN_EN['height_dim'] = 1
CONFIG_WGAN_HUMAN_EN["proportional_neuron"] = True # If sets to True then the neuron size in each layer is proportional to the snp length
CONFIG_WGAN_HUMAN_EN['hidden_dim1_g'] = 512
CONFIG_WGAN_HUMAN_EN['hidden_dim2_g'] = 1024
CONFIG_WGAN_HUMAN_EN['hidden_dim3_g'] = 2048
CONFIG_WGAN_HUMAN_EN['hidden_dim1_d'] = 2048
CONFIG_WGAN_HUMAN_EN['hidden_dim2_d'] = 1024
CONFIG_WGAN_HUMAN_EN['hidden_dim3_d'] = 512

if CONFIG_WGAN_HUMAN_EN["proportional_neuron"]:
    CONFIG_WGAN_HUMAN_EN['hidden_dim1_g'] = (3 * CONFIG_WGAN_HUMAN_EN['x_dim']) // 24
    CONFIG_WGAN_HUMAN_EN['hidden_dim2_g'] = (3 * CONFIG_WGAN_HUMAN_EN['x_dim']) // 12
    CONFIG_WGAN_HUMAN_EN['hidden_dim3_g'] = (3 * CONFIG_WGAN_HUMAN_EN['x_dim']) // 6
    CONFIG_WGAN_HUMAN_EN["hidden_dim1_d"] = (3 * CONFIG_WGAN_HUMAN_EN['x_dim']) // 8
    CONFIG_WGAN_HUMAN_EN["hidden_dim2_d"] = (3 * CONFIG_WGAN_HUMAN_EN['x_dim']) // 16
    CONFIG_WGAN_HUMAN_EN["hidden_dim3_d"] = (3 * CONFIG_WGAN_HUMAN_EN['x_dim']) // 32

# Training
CONFIG_WGAN_HUMAN_EN['activation'] = 'leaky_relu'
CONFIG_WGAN_HUMAN_EN['negative_slope'] = 0.05
CONFIG_WGAN_HUMAN_EN['optimizer'] = 'adam'
CONFIG_WGAN_HUMAN_EN['lr_g'] = 1e-4
CONFIG_WGAN_HUMAN_EN['lr_d'] = 1e-4
CONFIG_WGAN_HUMAN_EN['batch_size'] = 512
CONFIG_WGAN_HUMAN_EN['epochs'] = 1000
CONFIG_WGAN_HUMAN_EN['n_critic'] = 5
CONFIG_WGAN_HUMAN_EN['lambda_penalty'] = 10
CONFIG_WGAN_HUMAN_EN['step'] = 20
CONFIG_WGAN_HUMAN_EN['init_temp'] = 1
CONFIG_WGAN_HUMAN_EN['final_temp'] = 0.1

# Device
device = torch.device("cuda:2")