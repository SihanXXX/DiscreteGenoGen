import os
import sys
from pathlib import Path
import argparse 
import random
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from gan_model import GAN

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# Load Configuration
from gan_cow_ch14_configs import (CONFIG_GAN_COW_CH14,device)

# Load Data
parser = argparse.ArgumentParser(description="Train GAN model on genotype data")
parser.add_argument('--data_path', type=str, required=True, help='Path to training data folder')
args = parser.parse_args()
# Use the parsed arguments
data_path = args.data_path
# Validation dataset
val_bov_ch14 = pd.read_parquet(data_path + "val.parquet")

# Customized Dataset to process genotype data
class GenotypeDataset(Dataset):
    """
    Customized Dataset to load SNP genotypes
    """
    def __init__(self, data_path: str, type: str = "train", inject_noise: bool = True):
        self.data_path = data_path
        self.type = type
        self.inject_noise = inject_noise
        # Preprocess genotype data
        geno_df = pd.read_parquet(self.data_path + self.type + ".parquet")
        self.geno_tensor = self.one_hot_encode_genotype_data(geno_df)

    def __len__(self):
        return len(self.geno_tensor)

    def __getitem__(self, idx):
        geno = self.geno_tensor[idx]
        return geno

    def one_hot_encode_genotype_data(self, geno_df):
        """
        One-hot encoding genotype dataframe (and inject noise to stabilize training)

        Args:
            geno_df: dataframe of shape [Nb_Individuals, Nb_SNPs]
        
        Returns:
            geno_one_hot_tensor (torch.Tensor): A tensor of shape [Nb_Individuals, Nb_SNPs,3]
        """
        geno_one_hot_tensor = F.one_hot(torch.tensor(geno_df.values, dtype=torch.float32).long(), num_classes=3).float()
        if not self.inject_noise:
            return geno_one_hot_tensor
        else:
            noise = torch.empty_like(geno_one_hot_tensor).uniform_(*(0, 0.025))
            zero_mask = (geno_one_hot_tensor == 0).float()  # Mask for 0 elements
            one_mask = (geno_one_hot_tensor == 1).float()  # Mask for 1 elements
            noisy_tensor = geno_one_hot_tensor + zero_mask * noise # Add noise to elements with value 0
            noise_sum_per_vector = torch.sum(noise * zero_mask, dim=-1, keepdim=True)  # Sum of noise for 0 elements
            noisy_tensor -= one_mask * noise_sum_per_vector  # Reduce noise for the element with value 1
            return noisy_tensor

data = GenotypeDataset(data_path)
train_dataloader = DataLoader(data, batch_size=CONFIG_GAN_COW_CH14['batch_size'], shuffle=True, pin_memory=True)    

# Load Model
model = GAN(CONFIG_GAN_COW_CH14, device)
# Train Model
model.train(train_dataloader, val_bov_ch14, CONFIG_GAN_COW_CH14['latent_dim'], CONFIG_GAN_COW_CH14['epochs'],initial_temp=CONFIG_GAN_COW_CH14['init_temp'],final_temp=CONFIG_GAN_COW_CH14['final_temp'],step=CONFIG_GAN_COW_CH14['step'])