import torch

# Model & Training Parameters
sequence_length = 1771
batch_size = 256
learning_rate = 5e-4
num_epochs = 1000
encoder_dims = [1024, 512, 256]
decoder_dims = [256, 512, 1024]
latent_dim = 128
step = 20

# Device Configuration
device = torch.device("cuda:2")
