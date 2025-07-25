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
from wgan_human_multi_configs import (CONFIG_WGAN_HUMAN_MULTI,device)

# Load Data
parser = argparse.ArgumentParser(description="Train WGAN model on genotype data")
parser.add_argument('--data_path', type=str, required=True, help='Path to training data folder')
args = parser.parse_args()
# Use the parsed arguments
data_path = args.data_path
# Validation dataset
val_RG = pd.read_parquet(data_path + "val.parquet")
val_pheno = pd.read_parquet(data_path + "val_pheno.parquet")
val_pheno['Sex'] = val_pheno['Sex'].replace({2: 0})
if CONFIG_WGAN_HUMAN_MULTI['rescale_height']:
    val_pheno['Height'] = val_pheno['Height']/100 # cm to m
val_sex = torch.tensor(val_pheno['Sex'].values, dtype=torch.float32).unsqueeze(1)
val_height = torch.tensor(val_pheno['Height'].values, dtype=torch.float32).unsqueeze(1)

class GenotypeDataset(Dataset):
    """
    Customized Dataset to load SNP genotypes, sex and height
    """
    def __init__(self, data_path: str, snp_repr: str, rescale_height: bool, type: str = "train", inject_noise: bool = True):
        """Parameters:
            data_path (str): data .parquet file used for training
            snp_repr (str): snp representation of genotype dataset, choose between 'one-hot', 'count' and "freq" 
            rescale_height (bool): if we divide the height by 100, in otherwords from cm to m
            type (str): choose between train, test and val
            inject_noise (bool): inject noise to input or not
        """
        self.data_path = data_path
        self.snp_repr = snp_repr
        self.rescale_height = rescale_height
        self.type = type
        self.inject_noise = inject_noise
        # Preprocess genotype data
        geno_df = pd.read_parquet(self.data_path + self.type + ".parquet")
        if self.snp_repr == "one-hot":
            self.geno_tensor = self.one_hot_encode_genotype_data(geno_df)
        else:
            geno_df = self.preprocess_genotype_data(geno_df, snp_repr)
            self.geno_tensor = torch.tensor(geno_df.values, dtype=torch.float32)
        # Preprocess phenotype data
        pheno_df = pd.read_parquet(self.data_path + "pheno.parquet")
        pheno_df = self.preprocess_phenotype_data(pheno_df, rescale_height)
        # Convert DataFrame to torch tensors
        self.sex_tensor = torch.tensor(pheno_df['Sex'].values, dtype=torch.float32).unsqueeze(1)  # Add dimension
        self.height_tensor = torch.tensor(pheno_df['Height'].values, dtype=torch.float32).unsqueeze(1)  # Add dimension

    def __len__(self):
        return len(self.geno_tensor)

    def __getitem__(self, idx):
        geno = self.geno_tensor[idx]
        sex = self.sex_tensor[idx]
        height = self.height_tensor[idx]
        return geno, sex, height

    def preprocess_genotype_data(self, geno_df, snp_repr):
        if snp_repr == "count":
            # Map the values: 0 -> -1, 1 -> 0, 2 -> 1
            geno_mapped = geno_df.replace({0: -1, 1: 0, 2: 1})
        elif snp_repr == "freq":
            # Map the values: 0 -> 0, 1 -> 0.5, 2 -> 1
            geno_mapped = geno_df.replace({0: 0.0, 1: 0.5, 2: 1.0})
        else:
            raise ValueError("snp_repr should take a string value between count and freq")

        if self.inject_noise:
            # Inject small uniform noise
            noise = np.random.uniform(-0.025, 0.025, size=geno_mapped.shape)
            geno_mapped = geno_mapped + noise
            
        if snp_repr == "count":
            # Clip values to be within the range of -1 and 1
            geno_clipped = np.clip(geno_mapped, -1, 1)
        elif snp_repr == "freq":
            # Clip values to be within the range of 0 and 1
            geno_clipped = np.clip(geno_mapped, 0, 1)
    
        return geno_clipped

    def preprocess_phenotype_data(self, pheno_df, rescale_height):
        # Transform 2 -> 0 (binary encoding: 1 for men, 0 for women)
        pheno_df['Sex'] = pheno_df['Sex'].replace({2: 0})
        if rescale_height:
            pheno_df['Height'] = pheno_df['Height']/100

        return pheno_df

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
print("-> Loading data")
data = GenotypeDataset(data_path, CONFIG_WGAN_HUMAN_MULTI['snp_repr'], CONFIG_WGAN_HUMAN_MULTI['rescale_height'])
train_dataloader = DataLoader(data, batch_size=CONFIG_WGAN_HUMAN_MULTI['batch_size'], shuffle=True, pin_memory=True)    
# Create Model
model = WGAN_GP(CONFIG_WGAN_HUMAN_MULTI, device)
# Train Model
model.train(train_dataloader, val_RG, val_sex.to(device), val_height.to(device), CONFIG_WGAN_HUMAN_MULTI['latent_dim'], CONFIG_WGAN_HUMAN_MULTI['epochs'],initial_temp=CONFIG_WGAN_HUMAN_MULTI['init_temp'],final_temp=CONFIG_WGAN_HUMAN_MULTI['final_temp'],step=CONFIG_WGAN_HUMAN_MULTI['step'])