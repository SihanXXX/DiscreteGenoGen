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
from wgan_model import WGAN_GP

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# Load Configuration
from wgan_cow_all_configs import (CONFIG_WGAN_COW_ALL,device)

# Load Data
parser = argparse.ArgumentParser(description="Train WGAN model on genotype data")
parser.add_argument('--data_path', type=str, required=True, help='Path to training data folder')
args = parser.parse_args()
# Use the parsed arguments
data_path = args.data_path
# Validation dataset
val_RG = pd.read_parquet(data_path + "val.parquet")
val_pheno = pd.read_parquet(data_path + "val_pheno.parquet")
val_pheno_tensor = torch.tensor(val_pheno.values, dtype=torch.float32).unsqueeze(1)

class GenotypeDataset(Dataset):
    """
    Customized Dataset to load SNP genotypes and a trait
    """
    def __init__(self, data_path: str, type: str = "train", inject_noise: bool = True):
        slef.data_path = data_path
        self.type = type
        self.inject_noise = inject_noise
        # Preprocess genotype data
        geno_df = pd.read_parquet(self.data_path + self.type + ".parquet")
        self.geno_tensor = self.one_hot_encode_genotype_data(geno_df)
        # Preprocess phenotype data
        pheno_df = pd.read_parquet(self.data_path + "pheno.parquet")
        self.pheno_tensor = torch.tensor(pheno_df.values, dtype=torch.float32).unsqueeze(1)  # Add dimension

    def __len__(self):
        return len(self.geno_tensor)

    def __getitem__(self, idx):
        geno = self.geno_tensor[idx]
        pheno = self.pheno_tensor[idx]
        return geno, pheno

    def one_hot_encode_genotype_data(self, geno_df):
        """
        One-hot encoding genotype dataframe (and inject noise)

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

# Dataloader
data = GenotypeDataset(data_path)
train_dataloader = DataLoader(data, batch_size=CONFIG_WGAN_COW_ALL['batch_size'], shuffle=True, pin_memory=True)   

# Load Model
model = WGAN_GP(CONFIG_WGAN_COW_ALL, device)

# Train Model
model.train(train_dataloader,val_RG,val_pheno_tensor.to(device),CONFIG_WGAN_COW_ALL['latent_dim'], CONFIG_WGAN_COW_ALL['epochs'],initial_temp=CONFIG_WGAN_COW_ALL['init_temp'],final_temp=CONFIG_WGAN_COW_ALL['final_temp'],step=CONFIG_WGAN_COW_ALL['step'])