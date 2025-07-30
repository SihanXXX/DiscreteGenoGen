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
from dm_cow_all_configs import (CONFIG_DM_COW_ALL,device_dm)

# Customized Dataset
class SNPPCADataset(Dataset):
    """
    Customized Dataset to load SNP genotypes and phenotypes
    """
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

# Load Data
parser = argparse.ArgumentParser(description="Train DM model on genotype data")
parser.add_argument('--data_path', type=str, required=True, help='Path to training data folder')
args = parser.parse_args()
# Use the parsed arguments
data_path = args.data_path
train_data = np.load(data_path + "train.parquet")
pheno_train = pd.read_parquet(data_path + "pheno.parquet")
train_dataset = SNPPCADataset(train_data, pheno_train)
train_dataloader = DataLoader(train_dataset, batch_size=CONFIG_DM_COW_ALL["batch_size"], shuffle=True, pin_memory=True)

# Validation dataset
val_RG = pd.read_parquet(data_path + "val.parquet")
val_pheno = pd.read_parquet(data_path + "val_pheno.parquet")
val_pheno_tensor = torch.tensor(val_pheno.values, dtype=torch.float32).unsqueeze(1)

# Start training
num_batches = len(train_dataloader)
total_steps = CONFIG_DM_COW_ALL["num_epochs"] * num_batches
CONFIG_DM_COW_ALL["total_steps_lr_schedule"] = total_steps - CONFIG_DM_COW_ALL["warmup_period"]
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
diffusion_process = GaussianDiffusion(num_diffusion_timesteps=CONFIG_DM_COW_ALL["num_timesteps"], device = device_dm)
time_sampler = TimeSampler(diffusion_process.tmin, diffusion_process.tmax)
noise_predictor = NoisePredictor(CONFIG_DM_COW_ALL["snp_dim"],CONFIG_DM_COW_ALL["time_embedding_dim"],CONFIG_DM_COW_ALL["label_embedding_dim"],CONFIG_DM_COW_ALL["hidden_dim_1"],CONFIG_DM_COW_ALL["hidden_dim_2"],CONFIG_DM_COW_ALL["hidden_dim_3"],CONFIG_DM_COW_ALL["num_timesteps"]).to(device_dm)
model = DDPM(CONFIG_DM_COW_ALL["snp_dim"], diffusion_process, time_sampler, noise_predictor).to(device_dm)
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=CONFIG_DM_COW_ALL["optim_lr"])
# Use a cosine annealing scheduler with warm restarts.
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG_DM_COW_ALL["total_steps_lr_schedule"], eta_min=CONFIG_DM_COW_ALL["optim_lr_min"])
warmup_scheduler = warmup.LinearWarmup(optimizer, CONFIG_DM_COW_ALL["warmup_period"])

best_f1 = 0.0
for epoch in range(CONFIG_DM_COW_ALL["num_epochs"]):
    model.train()
    epoch_loss = 0.0    
    for batch_idx, (x0, label) in enumerate(train_dataloader):
        x0 = x0.to(device_dm)   
        label = label.to(device_dm) 
        optimizer.zero_grad() 
        loss = model.loss(x0, label) # Compute the loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= CONFIG_DM_COW_ALL["warmup_period"]:
                lr_scheduler.step()
    
    avg_loss = epoch_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    # print(f"Epoch {epoch+1}/{num_epochs} average loss: {avg_loss:.6f}, LR: {current_lr:.6e}")

    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{avg_loss:.6f},{current_lr:.6e}\n")
    
    if (epoch + 1) % CONFIG_DM_COW_ALL["checkpoint"] == 0:
        # generate AGs and compute the metrics
        metric_dir = os.path.join(log_dir, str(epoch+1))
        os.makedirs(metric_dir, exist_ok=True)
        fake_geno = pd.DataFrame(model.generate(val_pheno_tensor.to(device_dm)), columns=val_RG.columns, dtype=float)
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
    if (epoch + 1) == CONFIG_DM_COW_ALL["num_epochs"]:
        model_path = os.path.join(checkpoint_dir, "final_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at final epoch.")