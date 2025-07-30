import torch

# DM config 
CONFIG_DM_COW_ALL = dict()

# Data
CONFIG_DM_COW_ALL["snp_dim"] = 4819

# Architecture
CONFIG_DM_COW_ALL["num_timesteps"] = 1500
CONFIG_DM_COW_ALL["time_embedding_dim"] = 256
CONFIG_DM_COW_ALL["label_embedding_dim"] = 64
CONFIG_DM_COW_ALL["hidden_dim_1"] = 8192
CONFIG_DM_COW_ALL["hidden_dim_2"] = 8192
CONFIG_DM_COW_ALL["hidden_dim_3"] = 6144

# Training
CONFIG_DM_COW_ALL["batch_size"] = 4086
CONFIG_DM_COW_ALL["optim_lr"] = 3e-4
CONFIG_DM_COW_ALL["optim_lr_min"] = 1e-6 # 3e-12 have been used in some diffusion model implementations.  
CONFIG_DM_COW_ALL["num_epochs"] = 20000       
CONFIG_DM_COW_ALL["checkpoint"] = 100
CONFIG_DM_COW_ALL["warmup_period"] = 1000   

# Device
device_dm = torch.device("cuda:0")