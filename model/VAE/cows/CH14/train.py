import os
import sys
from pathlib import Path
import argparse 
import random
import time 
import datetime
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn.utils.spectral_norm as SN
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import VAE 

# Get the absolute path to the metric directory
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../../../"))
metric_dir = os.path.join(project_root, "metric")
# Add metric directory to sys.path
if metric_dir not in sys.path:
    sys.path.append(metric_dir)
# Now import the evaluator
from evaluator import train_evaluator

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# Loss function for VAE
def vae_loss(logits, x_int, mu, logvar):
    """
    Computes the VAE loss as the sum of reconstruction loss and KL divergence.
    
    logits: Tensor of shape (batch_size, sequence_length, 3) (decoder outputs)
    x_int: Tensor of shape (batch_size, sequence_length) with integer genotype labels {0,1,2}
    mu, logvar: Latent parameters from the encoder.
    """
    batch_size, seq_len, num_classes = logits.size()
    # Flatten logits to shape (batch_size * sequence_length, 3)
    logits_flat = logits.view(batch_size * seq_len, num_classes)
    # Flatten target labels to shape (batch_size * sequence_length)
    targets_flat = x_int.view(-1)
    # Reconstruction loss: cross-entropy loss summed over all positions
    reconstruction_loss = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='sum')
    # KL divergence: measures divergence from the standard normal distribution
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return reconstruction_loss + kl_loss


# Load Configuration
from configs import (
    sequence_length,
    batch_size,
    learning_rate,
    num_epochs,
    encoder_dims,
    decoder_dims,
    latent_dim,
    step,
    device
)

# Loading Data
print("-> Loading Training data")
RG = pd.read_pickle("/home/jovyan/workspace/Dataset/Train_Val_Test_Sets_Data_Total/typages_SNP_train.pkl").iloc[:,1:]
snp_info = pd.read_csv("/home/jovyan/data/position_variants_50Kp.txt", sep=" ", header=None)
snp_info.columns = ["Index", "Chromosome", "SNP_ID", "Position_MB"]
# Calculate variance for each SNP (column)
variances = RG.var(axis=0)
# Filter SNPs that have variance above the threshold
high_variance_snps = variances[variances > 0.001].index
# Return filtered DataFrame
filtered_RG = RG[high_variance_snps]
bov_ch14 = filtered_RG.iloc[:, 28784:30555]
sampled_bov_ch14 = bov_ch14.sample(n=5000, random_state=42)
geno_tensor = torch.tensor(bov_ch14.values).long()
geno_dataset = TensorDataset(geno_tensor)
train_dataloader = DataLoader(geno_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Load Model and Optimizer
model = VAE(sequence_length, latent_dim, encoder_dims, decoder_dims).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set up logs and checkpoints
log_dir = './logs'
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join(log_dir, current_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    print("Log Directory for current experiment '%s' created" % log_dir)
log_file = os.path.join(log_dir, "training_loss_log.txt")
other_metrics_file = os.path.join(log_dir, "precision_recall_fst.txt")
with open(log_file, "w") as f1:
    f1.write("Epoch\t Loss\n")
with open(other_metrics_file, "w") as f2:
    f2.write("Epoch\tPrecision\tRecall\tFST\n")

# Define checkpoints path
checkpoint_dir = "./checkpoints"
checkpoint_dir = os.path.join(checkpoint_dir, current_time)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Checkpoint Directory for current experiment '%s' created" % checkpoint_dir)

# Training 
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_dataloader)  
    for batch in train_dataloader:
        # x_int has shape: (batch_size, sequence_length) with integer values
        x_int = batch[0].to(device)
        # Convert x_int to one-hot encoding for the VAE input
        x_onehot = torch.nn.functional.one_hot(x_int, num_classes=3).float()
        
        optimizer.zero_grad()
        logits, mu, logvar = model(x_onehot)
        loss = vae_loss(logits, x_int, mu, logvar)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    with open(log_file, "a") as f:
        f.write(f"{epoch+1}\t{avg_loss:.6f}\n")

    if (epoch+1)%step == 0:
        metric_dir = os.path.join(log_dir, str(epoch+1))
        os.makedirs(metric_dir, exist_ok=True)
        fake_geno = pd.DataFrame(model.generate(sampled_bov_ch14.shape[0]), columns=sampled_bov_ch14.columns, dtype=float)
        metrics_result = train_evaluator(sampled_bov_ch14, fake_geno, ["precision_recall","fixation_index","pca","allele_freq","geno_freq"], "real", "syn", metric_dir+"/")
        # Write metric
        with open(other_metrics_file, "a") as f:
            f.write(f"{epoch+1}\t{metrics_result['precision']:.4f}\t{metrics_result['recall']:.4f}\t{metrics_result['fixation_index']:.4f}\n")
        
                        
        gen_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
        torch.save(model.decoder.state_dict(), gen_path)
        print(f"Saved decoder at epoch {epoch + 1}")