import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

#device = torch.device("cuda:0")

def create_dataloader(X, y, batch_size=64):
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) 
            layers.append(nn.ReLU())  
            layers.append(nn.Dropout(dropout))  
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1)) 
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super(MLPRegressor, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) 
            layers.append(nn.ReLU())  
            layers.append(nn.Dropout(dropout))  
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1)) 
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#Device
device = torch.device("cuda:0")
# Set Hyperparameters
hidden_dims = [512, 256, 128]  # Decreasing layer sizes
dropout = 0.2  
learning_rate = 1e-4
lr_weight_decay = 1e-5
batch_size = 256
num_epochs = 20

# Create a file to store results
result_file = "result_cows_ch14.txt"
with open(result_file, "w") as f:
    f.write("MLP Regression Test Results\n")
    f.write("=" * 40 + "\n")



