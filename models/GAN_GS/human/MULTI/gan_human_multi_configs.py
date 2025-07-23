import torch

# GAN config 
CONFIG_GAN_HUMAN_MULTI = dict()

# Data
CONFIG_GAN_HUMAN_MULTI['x_dim'] = 42409
CONFIG_GAN_HUMAN_MULTI['snp_repr'] = "one-hot"
CONFIG_GAN_HUMAN_MULTI['rescale_height'] = True

# Architecture
CONFIG_GAN_HUMAN_MULTI['latent_dim'] = 256
CONFIG_GAN_HUMAN_MULTI["proportional_neuron"] = True # If sets to True then the neuron size in each layer is proportional to the snp length
CONFIG_GAN_HUMAN_MULTI['hidden_dim1_g'] = 512
CONFIG_GAN_HUMAN_MULTI['hidden_dim2_g'] = 1024
CONFIG_GAN_HUMAN_MULTI['hidden_dim3_g'] = 4096
CONFIG_GAN_HUMAN_MULTI['hidden_dim1_d'] = 4096
CONFIG_GAN_HUMAN_MULTI['hidden_dim2_d'] = 1024
CONFIG_GAN_HUMAN_MULTI['hidden_dim3_d'] = 512

if CONFIG_GAN_HUMAN_MULTI["proportional_neuron"]:
    CONFIG_GAN_HUMAN_MULTI['hidden_dim1_g'] = (3 * CONFIG_GAN_HUMAN_MULTI['x_dim']) // 128
    CONFIG_GAN_HUMAN_MULTI['hidden_dim2_g'] = (3 * CONFIG_GAN_HUMAN_MULTI['x_dim']) // 64
    CONFIG_GAN_HUMAN_MULTI['hidden_dim3_g'] = (3 * CONFIG_GAN_HUMAN_MULTI['x_dim']) // 32
    CONFIG_GAN_HUMAN_MULTI["hidden_dim1_d"] = (3 * CONFIG_GAN_HUMAN_MULTI['x_dim']) // 32
    CONFIG_GAN_HUMAN_MULTI["hidden_dim2_d"] = (3 * CONFIG_GAN_HUMAN_MULTI['x_dim']) // 64
    CONFIG_GAN_HUMAN_MULTI["hidden_dim3_d"] = (3 * CONFIG_GAN_HUMAN_MULTI['x_dim']) // 128

# Training
CONFIG_GAN_HUMAN_MULTI['activation'] = 'leaky_relu'
CONFIG_GAN_HUMAN_MULTI['negative_slope'] = 0.05
CONFIG_GAN_HUMAN_MULTI['optimizer'] = 'adam'
CONFIG_GAN_HUMAN_MULTI['lr_g'] = 1e-4
CONFIG_GAN_HUMAN_MULTI['lr_d'] = 1e-5
CONFIG_GAN_HUMAN_MULTI['batch_size'] = 128
CONFIG_GAN_HUMAN_MULTI['epochs'] = 1000
CONFIG_GAN_HUMAN_MULTI['step'] = 10
CONFIG_GAN_HUMAN_MULTI['init_temp'] = 1
CONFIG_GAN_HUMAN_MULTI['final_temp'] = 0.1

# Device
device = torch.device("cuda:0")