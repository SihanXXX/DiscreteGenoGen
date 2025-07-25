import torch

CONFIG_WGAN_HUMAN_MULTI= dict()

# Data
CONFIG_WGAN_HUMAN_MULTI['x_dim'] = 42409
CONFIG_WGAN_HUMAN_MULTI['snp_repr'] = "one-hot"
CONFIG_WGAN_HUMAN_MULTI['rescale_height'] = True

# Architecture
CONFIG_WGAN_HUMAN_MULTI['latent_dim'] = 256
CONFIG_WGAN_HUMAN_MULTI['sex_dim'] = 1
CONFIG_WGAN_HUMAN_MULTI['height_dim'] = 1
CONFIG_WGAN_HUMAN_MULTI["proportional_neuron"] = True # If sets to True then the neuron size in each layer is proportional to the snp length
CONFIG_WGAN_HUMAN_MULTI['hidden_dim1_g'] = 512
CONFIG_WGAN_HUMAN_MULTI['hidden_dim2_g'] = 1024
CONFIG_WGAN_HUMAN_MULTI['hidden_dim3_g'] = 4096
CONFIG_WGAN_HUMAN_MULTI['hidden_dim1_d'] = 4096
CONFIG_WGAN_HUMAN_MULTI['hidden_dim2_d'] = 1024
CONFIG_WGAN_HUMAN_MULTI['hidden_dim3_d'] = 512

if CONFIG_WGAN_HUMAN_MULTI["proportional_neuron"]:
    CONFIG_WGAN_HUMAN_MULTI['hidden_dim1_g'] = (3 * CONFIG_WGAN_HUMAN_MULTI['x_dim']) // 128
    CONFIG_WGAN_HUMAN_MULTI['hidden_dim2_g'] = (3 * CONFIG_WGAN_HUMAN_MULTI['x_dim']) // 64
    CONFIG_WGAN_HUMAN_MULTI['hidden_dim3_g'] = (3 * CONFIG_WGAN_HUMAN_MULTI['x_dim']) // 32
    CONFIG_WGAN_HUMAN_MULTI["hidden_dim1_d"] = (3 * CONFIG_WGAN_HUMAN_MULTI['x_dim']) // 32
    CONFIG_WGAN_HUMAN_MULTI["hidden_dim2_d"] = (3 * CONFIG_WGAN_HUMAN_MULTI['x_dim']) // 64
    CONFIG_WGAN_HUMAN_MULTI["hidden_dim3_d"] = (3 * CONFIG_WGAN_HUMAN_MULTI['x_dim']) // 128

# Training
CONFIG_WGAN_HUMAN_MULTI['activation'] = 'leaky_relu'
CONFIG_WGAN_HUMAN_MULTI['negative_slope'] = 0.05
CONFIG_WGAN_HUMAN_MULTI['optimizer'] = 'adam'
CONFIG_WGAN_HUMAN_MULTI['lr_g'] = 1e-4
CONFIG_WGAN_HUMAN_MULTI['lr_d'] = 1e-4
CONFIG_WGAN_HUMAN_MULTI['batch_size'] = 128
CONFIG_WGAN_HUMAN_MULTI['epochs'] = 1000
CONFIG_WGAN_HUMAN_MULTI['n_critic'] = 5
CONFIG_WGAN_HUMAN_MULTI['lambda_penalty'] = 10
CONFIG_WGAN_HUMAN_MULTI['step'] = 20
CONFIG_WGAN_HUMAN_MULTI['init_temp'] = 1
CONFIG_WGAN_HUMAN_MULTI['final_temp'] = 0.1

# Device
device = torch.device("cuda:0")