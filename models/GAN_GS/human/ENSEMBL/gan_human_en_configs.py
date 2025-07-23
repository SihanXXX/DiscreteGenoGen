import torch

# GAN config 
CONFIG_GAN_HUMAN_EN = dict()

# Data
CONFIG_GAN_HUMAN_EN['x_dim'] = 3493
CONFIG_GAN_HUMAN_EN['snp_repr'] = "one-hot"
CONFIG_GAN_HUMAN_EN['rescale_height'] = True

# Architecture
CONFIG_GAN_HUMAN_EN['latent_dim'] = 128
CONFIG_GAN_HUMAN_EN['sex_dim'] = 1
CONFIG_GAN_HUMAN_EN['height_dim'] = 1
CONFIG_GAN_HUMAN_EN["proportional_neuron"] = True # If sets to True then the neuron size in each layer is proportional to the snp length
CONFIG_GAN_HUMAN_EN['hidden_dim1_g'] = 512
CONFIG_GAN_HUMAN_EN['hidden_dim2_g'] = 1024
CONFIG_GAN_HUMAN_EN['hidden_dim3_g'] = 2048
CONFIG_GAN_HUMAN_EN['hidden_dim1_d'] = 2048
CONFIG_GAN_HUMAN_EN['hidden_dim2_d'] = 1024
CONFIG_GAN_HUMAN_EN['hidden_dim3_d'] = 512

if CONFIG_GAN_HUMAN_EN["proportional_neuron"]:
    CONFIG_GAN_HUMAN_EN['hidden_dim1_g'] = (3 * CONFIG_GAN_HUMAN_EN['x_dim']) // 24
    CONFIG_GAN_HUMAN_EN['hidden_dim2_g'] = (3 * CONFIG_GAN_HUMAN_EN['x_dim']) // 12
    CONFIG_GAN_HUMAN_EN['hidden_dim3_g'] = (3 * CONFIG_GAN_HUMAN_EN['x_dim']) // 6
    CONFIG_GAN_HUMAN_EN["hidden_dim1_d"] = (3 * CONFIG_GAN_HUMAN_EN['x_dim']) // 8
    CONFIG_GAN_HUMAN_EN["hidden_dim2_d"] = (3 * CONFIG_GAN_HUMAN_EN['x_dim']) // 16
    CONFIG_GAN_HUMAN_EN["hidden_dim3_d"] = (3 * CONFIG_GAN_HUMAN_EN['x_dim']) // 32

# Training
CONFIG_GAN_HUMAN_EN['activation'] = 'leaky_relu'
CONFIG_GAN_HUMAN_EN['negative_slope'] = 0.05
CONFIG_GAN_HUMAN_EN['optimizer'] = 'adam'
CONFIG_GAN_HUMAN_EN['lr_g'] = 1e-4
CONFIG_GAN_HUMAN_EN['lr_d'] = 1e-4
CONFIG_GAN_HUMAN_EN['batch_size'] = 512
CONFIG_GAN_HUMAN_EN['epochs'] = 1000
CONFIG_GAN_HUMAN_EN['step'] = 10
CONFIG_GAN_HUMAN_EN['init_temp'] = 1
CONFIG_GAN_HUMAN_EN['final_temp'] = 0.1

# Device Configuration
device = torch.device("cuda:2")