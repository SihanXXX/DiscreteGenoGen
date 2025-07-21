import argparse
import os
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

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    model.to(device)
    best_val_loss = float("inf")
    best_model = None
    
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred_val = model(X_val).squeeze()
                loss = criterion(y_pred_val, y_val)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print("train loss: " + str(train_loss) + ", val loss: " + str(val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()  # Save best model state

    model.load_state_dict(best_model)  # Load best model
    return model, best_val_loss

def objective(trial, X_train, y_train, X_val, y_val):
    hidden_dims = [
        trial.suggest_int("hidden_dim1", 128, 1024, step=128),
        trial.suggest_int("hidden_dim2", 64, 512, step=64),
        trial.suggest_int("hidden_dim3", 32, 256, step=32),
    ]
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    model = MLPRegressor(input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    train_loader = create_dataloader(X_train, y_train, batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

    # Store best model state_dict in global if this is the best trial
    if "loss" not in best_trained_model or val_loss < best_trained_model["loss"]:
        best_trained_model["loss"] = val_loss
        best_trained_model["model_state"] = model.state_dict()
        best_trained_model["params"] = trial.params
    
    return val_loss


#Device
device = torch.device("cuda:0")

# Starting Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP on genotype data for phenotype prediction")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--result_file', type=str, required=True, help='Path to save the result output file')
    args = parser.parse_args()

    result_file = args.result_file
    path_training_dataset = args.data_path
    
    # result file
    with open(result_file, "w") as f:
        f.write("MLP Results\n")
        f.write("=" * 40 + "\n")

    # Validation and Test set
    X_val = pd.read_parquet(path_training_dataset + "val.parquet")
    y_val = pd.read_parquet(path_training_dataset + "val_pheno.parquet")
    X_test = pd.read_parquet(path_training_dataset + "test.parquet")
    y_test = pd.read_parquet(path_training_dataset + "test_pheno.parquet")
    
    # Train on Real dataset
    X_train = pd.read_parquet(path_training_dataset + "real.parquet")
    y_train = pd.read_parquet(path_training_dataset + "pheno.parquet")

    best_trained_model = {}
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    print("Best parameters:", study.best_params)
    # Recover Best Model
    best_model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_dims=[
            best_trained_model["params"]["hidden_dim1"],
            best_trained_model["params"]["hidden_dim2"],
            best_trained_model["params"]["hidden_dim3"]
        ],
        dropout=best_trained_model["params"]["dropout"]
    ).to(device)
    best_model.load_state_dict(best_trained_model["model_state"])
    best_model.eval()  # Set to evaluation mode
    y_pred_test = best_model(torch.tensor(X_test.values, dtype=torch.float32).to(device)).cpu().detach().numpy().squeeze()
    mse = mean_squared_error(y_test, y_pred_test)
    corr = pearsonr(y_test.squeeze(), y_pred_test)[0]
    # Save results to file
    with open(result_file, "a") as f:
        f.write(f"Dataset Real:\nMSE: {mse:.4f}\nPearson Correlation: {corr:.4f}\n{'-'*30}\n")

    print(f"Dataset Real - MSE: {mse:.4f}, Pearson Correlation: {corr:.4f}")

    # Train on Synthetic Dataset
    X_train = pd.read_parquet(path_training_dataset + "syn.parquet")
    y_train = pd.read_parquet(path_training_dataset + "pheno.parquet")
    
    best_trained_model = {}
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    print("Best parameters:", study.best_params)
    # Recover Best Model
    best_model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden_dims=[
            best_trained_model["params"]["hidden_dim1"],
            best_trained_model["params"]["hidden_dim2"],
            best_trained_model["params"]["hidden_dim3"]
        ],
        dropout=best_trained_model["params"]["dropout"]
    ).to(device)
    best_model.load_state_dict(best_trained_model["model_state"])
    best_model.eval()  # Set to evaluation mode
    y_pred_test = best_model(torch.tensor(X_test.values, dtype=torch.float32).to(device)).cpu().detach().numpy().squeeze()
    mse = mean_squared_error(y_test, y_pred_test)
    corr = pearsonr(y_test.squeeze(), y_pred_test)[0]
    # Save results to file
    with open(result_file, "a") as f:
        f.write(f"Dataset Syn:\nMSE: {mse:.4f}\nPearson Correlation: {corr:.4f}\n{'-'*30}\n")

    print(f"Dataset Syn - MSE: {mse:.4f}, Pearson Correlation: {corr:.4f}")