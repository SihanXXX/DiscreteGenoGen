import torch

# Model & Training Parameters
sequence_length = 42409
batch_size = 3800
learning_rate = 6e-4
num_epochs = 1000
encoder_dims = [4096, 2048, 2048, 2048, 1024, 1024, 1024, 512]
decoder_dims = [512, 1024, 1024, 1024,2048, 2048, 2048, 4096]
latent_dim = 256
step = 50

# Device Configuration
device = torch.device("cuda:1")