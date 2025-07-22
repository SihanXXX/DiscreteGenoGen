import time 
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as SN

# Load Configuration
from vae_human_multi_configs import (
    device
)

# Xavier Inits
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# Residual Block that can be integrated in Encoder or Decoder
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, negative_slope = 0.05):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),  
            nn.LeakyReLU(negative_slope),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim)  
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        residual = x
        x = self.block(x)
        return self.leaky_relu(x + residual)

# Helper function to build layers from a list of hidden dimensions.
# If three consecutive numbers are the same, they are replaced by a ResidualBlock.
def build_layers(input_dim, dims_list, negative_slope=0.05):
    """
    Builds an nn.Sequential block from a list of hidden dimensions.
    
    If the list contains three consecutive identical values, those layers are
    replaced by a ResidualBlock operating at that dimension.
    
    Each standard layer is composed of:
      Linear -> BatchNorm1d -> LeakyReLU(negative_slope=negative_slope)
    
    Parameters:
      - input_dim (int): the size of the input.
      - dims_list (list of int): the list of desired hidden dimensions.
      - negative_slope (float): negative slope for the LeakyReLU.
    
    Returns:
      - A tuple (layers, output_dim) where:
          * layers is an nn.Sequential block,
          * output_dim is the final output dimension.
    """
    layers = []
    current_dim = input_dim
    i = 0
    while i < len(dims_list):
        # Check for three consecutive identical values.
        if i + 2 < len(dims_list) and dims_list[i] == dims_list[i+1] == dims_list[i+2]:
            target_dim = dims_list[i]
            # Adjust dimension if needed.
            if current_dim != target_dim:
                layers.append(nn.Linear(current_dim, target_dim))
                layers.append(nn.BatchNorm1d(target_dim))
                layers.append(nn.LeakyReLU(negative_slope=negative_slope))
                current_dim = target_dim
            layers.append(ResidualBlock(target_dim, negative_slope=negative_slope))
            i += 3  # Skip the next two since they're part of the residual block.
        else:
            target_dim = dims_list[i]
            layers.append(nn.Linear(current_dim, target_dim))
            layers.append(nn.BatchNorm1d(target_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            current_dim = target_dim
            i += 1
    return nn.Sequential(*layers), current_dim

class VAE(nn.Module):
    def __init__(self, sequence_length, latent_dim, encoder_hidden_dims, decoder_hidden_dims, negative_slope=0.05):
        """
        Parameters:
            sequence_length (int): Length of the genotype sequence.
            latent_dim (int): Dimension of the latent space.
            encoder_hidden_dims (list of int): List of hidden layer sizes for the encoder.
            decoder_hidden_dims (list of int): List of hidden layer sizes for the decoder.
            negative_slope (float): Negative slope for LeakyReLU.
            
        Input data is assumed to be one-hot encoded with shape:
            (batch_size, sequence_length, 3)
        """
        super(VAE, self).__init__()
        self.sequence_length = sequence_length
        self.input_dim = sequence_length * 3
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        self.encoder, encoder_out_dim = build_layers(self.input_dim, encoder_hidden_dims, negative_slope)
        self.encoder.apply(init_weights)
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)
        
        # --- Decoder ---
        decoder_layers, decoder_out_dim = build_layers(latent_dim, decoder_hidden_dims, negative_slope)
        final_layer = nn.Linear(decoder_out_dim, self.input_dim)
        self.decoder = nn.Sequential(decoder_layers, final_layer)
        self.decoder.apply(init_weights)

    def encode(self, x):
        # x: (batch_size, sequence_length, 3)
        x_flat = x.view(x.size(0), -1)  # Flatten to (batch_size, input_dim)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        logits_flat = self.decoder(z)
        logits = logits_flat.view(-1, self.sequence_length, 3)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def generate(self, nb, batch_size=1000):
        all_generated_samples = []
        self.eval()
        # Loop through mini-batches to generate the required number of samples
        for start_idx in range(0, nb, batch_size):
            end_idx = min(start_idx + batch_size, nb)  # Ensure we don't go beyond nb
            current_batch_size = end_idx - start_idx
            with torch.no_grad():
                z = torch.randn(current_batch_size, self.latent_dim).to(device)
                logits_gen = self.decode(z)  # Shape: (num_samples_to_generate, sequence_length, num_classes)
                generated_sequences = torch.argmax(logits_gen, dim=-1).cpu().numpy()  # Shape: (num_samples_to_generate, sequence_length)

            all_generated_samples.append(generated_sequences)
        # Concatenate all batches into one large array
        converted_AG = np.concatenate(all_generated_samples, axis=0)
        
        return converted_AG