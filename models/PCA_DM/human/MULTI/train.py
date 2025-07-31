import os
import sys
import time 
import datetime
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
import pytorch_warmup as warmup
from dm_model import GaussianDiffusion, TimeSampler, NoisePredictor, DDPM

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# Get the absolute path to the metric directory
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../../../"))
metric_dir = os.path.join(project_root, "metric")
# Add metric directory to sys.path
if metric_dir not in sys.path:
    sys.path.append(metric_dir)
# Now import the evaluator
from evaluator import train_evaluator

# Load Configuration
from dm_human_multi_configs import (CONFIG_DM_HUMAN_MULTI,device_dm)

# Function to generate the latent representation then reconstruct the genotype
def generate_reconstruct(diffusion_model, sex_tensor, height_tensor):
    # Reconstruction from PCA space to the original data space
    all_reconstructed_data = np.empty((sex_tensor.shape[0], 0))
    dims = [6104, 5149, 4438, 3078]
    cumdims = np.cumsum(dims)  # This gives [6104, 11253, 15691, 18769]
    start_indices = np.insert(cumdims, 0, 0)[:-1]  # This gives [0, 6104, 11253, 15691]
    # Generate genotype data in latent space
    diffusion_model.eval()
    with torch.no_grad():
        AG_PCA = np.array(diffusion_model.sample(sex_tensor.shape[0], sex_tensor, height_tensor).cpu())
    AG_PCA_slices = [AG_PCA[:, start:stop] for start, stop in zip(start_indices, cumdims)]

    cpt = 1
    for chr in [3,6,12,17]:
        result_folder = "../../../../pca/pca_data/ukb/multi/ch"+str(chr) # path where the pca component and mean are stored
        pca_components = np.load(result_folder + "/pca_components.npy")
        pca_mean = np.load(result_folder + "/pca_mean.npy")
        reconstructed_data = np.dot(AG_PCA_slices[cpt-1], pca_components) + pca_mean
        reconstructed_data = np.rint(reconstructed_data).clip(0, 2)  
        all_reconstructed_data = np.concatenate((all_reconstructed_data, reconstructed_data), axis=1)
        cpt = cpt + 1
    
    return all_reconstructed_data

# Customized Dataset Class
class SNPPCADataset(Dataset):
    def __init__(self, geno, pheno):
        self.geno = geno
        self.sex = pheno["Sex"]
        self.height = pheno["Height"]

    def __len__(self):
        return len(self.geno)

    def __getitem__(self, idx):
        geno = torch.tensor(self.geno[idx], dtype=torch.float32)
        sex = torch.tensor(self.sex[idx], dtype=torch.float32)
        height = torch.tensor(self.height[idx], dtype=torch.float32)
        return geno, sex, height

# Load Data
parser = argparse.ArgumentParser(description="Train DM model on genotype data")
parser.add_argument('--data_path', type=str, required=True, help='Path to training data folder')
args = parser.parse_args()
# Use the parsed arguments
data_path = args.data_path
# Training Dataset
train_data = pd.read_parquet(data_path + "train.parquet")
pheno_train = pd.read_parquet(data_path + "pheno.parquet")
pheno_train['Sex'] = pheno_train['Sex'].replace({2: 0})
pheno_train['Height'] = pheno_train['Height']/100
# Validation dataset
val_RG = pd.read_parquet(data_path + "val.parquet")
val_pheno = pd.read_parquet(data_path + "val_pheno.parquet")
val_pheno['Sex'] = val_pheno['Sex'].replace({2: 0})
if CONFIG_WGAN_HUMAN_EN['rescale_height']:
    val_pheno['Height'] = val_pheno['Height']/100 # cm to m
val_sex = torch.tensor(val_pheno['Sex'].values, dtype=torch.float32).unsqueeze(1)
val_height = torch.tensor(val_pheno['Height'].values, dtype=torch.float32).unsqueeze(1)
# PCA components
pca_components = np.load("../../../../pca_data/human/ENSEMBL/var0.9/pca_components.npy")
pca_mean = np.load("../../../../pca_data/human/ENSEMBL/var0.9/pca_mean.npy")

train_dataset = SNPPCADataset(train_data, pheno_train)
train_dataloader = DataLoader(train_dataset, batch_size=CONFIG_DM_HUMAN_MULTI["batch_size"], shuffle=True, pin_memory=True)

# Configuration
num_batches = len(train_dataloader)
total_steps = CONFIG_DM_HUMAN_MULTI["num_epochs"] * num_batches
CONFIG_DM_HUMAN_MULTI["total_steps_lr_schedule"] = total_steps - CONFIG_DM_HUMAN_MULTI["warmup_period"]

# Log
log_dir = './logs'
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join(log_dir, current_time)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print("Log Directory for current experiment '%s' created" % log_dir)
log_file = os.path.join(log_dir, "training_loss_log.txt")
with open(log_file, "w") as f:
    f.write("Epoch,AverageLoss,LearningRate\n")
other_metrics_file = os.path.join(log_dir, "precision_recall_fst.txt")
with open(other_metrics_file, "w") as f1:
    f1.write("Epoch\tPrecision\tRecall\tFST\n")
# Checkpoint
checkpoint_dir = "./checkpoints"
checkpoint_dir = os.path.join(checkpoint_dir, current_time)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print("Checkpoint Directory for current experiment '%s' created" % checkpoint_dir)

# Create Models
diffusion_process = GaussianDiffusion(num_diffusion_timesteps=CONFIG_DM_HUMAN_MULTI["num_timesteps"], device = device_dm)
time_sampler = TimeSampler(diffusion_process.tmin, diffusion_process.tmax)
noise_predictor = NoisePredictor(CONFIG_DM_HUMAN_MULTI["snp_dim"], CONFIG_DM_HUMAN_MULTI["time_embedding_dim"], CONFIG_DM_HUMAN_MULTI["sex_embedding_dim"], CONFIG_DM_HUMAN_MULTI["height_embedding_dim"],CONFIG_DM_HUMAN_MULTI["hidden_dim_1"],CONFIG_DM_HUMAN_MULTI["hidden_dim_2"],CONFIG_DM_HUMAN_MULTI["hidden_dim_3"],CONFIG_DM_HUMAN_MULTI["num_timesteps"]).to(device_dm)
model = DDPM(CONFIG_DM_HUMAN_MULTI["snp_dim"], diffusion_process, time_sampler, noise_predictor).to(device_dm)
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=CONFIG_DM_HUMAN_MULTI["optim_lr"])
# Use a cosine annealing scheduler with warm restarts.
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG_DM_HUMAN_MULTI["total_steps_lr_schedule"], eta_min=CONFIG_DM_HUMAN_MULTI["optim_lr_min"])
warmup_scheduler = warmup.LinearWarmup(optimizer, CONFIG_DM_HUMAN_MULTI["warmup_period"])

best_f1 = 0.0
for epoch in range(CONFIG_DM_HUMAN_MULTI["num_epochs"]):
    model.train()
    epoch_loss = 0.0    
    for batch_idx, (geno, sex, height) in enumerate(train_dataloader):
        geno = geno.to(device_dm)   
        sex = sex.to(device_dm) 
        height = height.to(device_dm) 
        optimizer.zero_grad() 
        loss = model.loss(geno, sex, height) # Compute the loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= CONFIG_DM_HUMAN_MULTI["warmup_period"]:
                lr_scheduler.step()
    
    avg_loss = epoch_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']

    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{avg_loss:.6f},{current_lr:.6e}\n")
    
    if (epoch + 1) % CONFIG_DM_HUMAN_MULTI["checkpoint"] == 0:
        # generate AGs and compute the metrics
        metric_dir = os.path.join(log_dir, str(epoch+1))
        os.makedirs(metric_dir, exist_ok=True)
        fake_geno = pd.DataFrame(generate_reconstruct(model, val_sex.to(device_dm), val_height.to(device_dm)), columns=val_RG.columns, dtype=float)
        metrics_result = train_evaluator(val_RG, fake_geno, ["precision_recall","fixation_index","pca","geno_freq"], "real", "syn", metric_dir+"/")
        if (metrics_result['precision'] + metrics_result['recall']) != 0:
            f1_score = 2 * (metrics_result['precision'] * metrics_result['recall']) / (metrics_result['precision'] + metrics_result['recall'])
        else:
            f1_score = 0
        # Write metric
        with open(other_metrics_file, "a") as f:
            f.write(f"{epoch+1}\t{metrics_result['precision']:.6f}\t{metrics_result['recall']:.6f}\t{metrics_result['fixation_index']:.6f}\n")

        if f1_score >= best_f1:  # Save the model only if the F1 score improved
            best_f1 = f1_score
            model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved at epoch {epoch+1} with F1 score {f1_score:.6f}")
            
    # Also save the final model 
    if (epoch + 1) == CONFIG_DM_HUMAN_MULTI["num_epochs"]:
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at final epoch.")