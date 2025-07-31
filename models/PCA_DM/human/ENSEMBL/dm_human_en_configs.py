import torch

# DM config 
CONFIG_DM_HUMAN_EN = dict()

# Data
CONFIG_DM_HUMAN_EN["snp_dim"] = 2026

# Architecture
CONFIG_DM_HUMAN_EN["num_timesteps"] = 1200
CONFIG_DM_HUMAN_EN["time_embedding_dim"] = 128
CONFIG_DM_HUMAN_EN["sex_embedding_dim"] = 16
CONFIG_DM_HUMAN_EN["height_embedding_dim"] = 32
CONFIG_DM_HUMAN_EN["hidden_dim_1"] = 4096
CONFIG_DM_HUMAN_EN["hidden_dim_2"] = 4096
CONFIG_DM_HUMAN_EN["hidden_dim_3"] = 2048

# Training
CONFIG_DM_HUMAN_EN["batch_size"] = 20480
CONFIG_DM_HUMAN_EN["optim_lr"] = 3e-4
CONFIG_DM_HUMAN_EN["optim_lr_min"] = 1e-6 # 3e-12 have been used in some diffusion model implementations.  
CONFIG_DM_HUMAN_EN["num_epochs"] = 20000       
CONFIG_DM_HUMAN_EN["checkpoint"] = 100
CONFIG_DM_HUMAN_EN["warmup_period"] = 1000   

# Device
device_dm = torch.device("cuda:1")