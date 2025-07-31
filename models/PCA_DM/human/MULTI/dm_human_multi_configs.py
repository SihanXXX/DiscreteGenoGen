import torch

# DM config 
CONFIG_DM_HUMAN_MULTI = dict()

# Data
CONFIG_DM_HUMAN_MULTI["snp_dim"] = 18769

# Architecture
CONFIG_DM_HUMAN_MULTI["num_timesteps"] = 1200
CONFIG_DM_HUMAN_MULTI["time_embedding_dim"] = 256
CONFIG_DM_HUMAN_MULTI["sex_embedding_dim"] = 16
CONFIG_DM_HUMAN_MULTI["height_embedding_dim"] = 32
CONFIG_DM_HUMAN_MULTI["hidden_dim_1"] = 12288
CONFIG_DM_HUMAN_MULTI["hidden_dim_2"] = 16384
CONFIG_DM_HUMAN_MULTI["hidden_dim_3"] = 16384

# Training
CONFIG_DM_HUMAN_MULTI["batch_size"] = 8096
CONFIG_DM_HUMAN_MULTI["optim_lr"] = 3e-4
CONFIG_DM_HUMAN_MULTI["optim_lr_min"] = 1e-6 # 3e-12 have been used in some diffusion model implementations.  
CONFIG_DM_HUMAN_MULTI["num_epochs"] = 20000       
CONFIG_DM_HUMAN_MULTI["checkpoint"] = 100
CONFIG_DM_HUMAN_MULTI["warmup_period"] = 1000   

# Device
device_dm = torch.device("cuda:1")