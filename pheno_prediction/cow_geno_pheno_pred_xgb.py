import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import optuna

# Set XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'eta': 0.1,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'min_child_weight': 1,
    'eval_metric': 'rmse'
}

# Define Optuna objective function (Now takes training & validation sets as input)
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 10.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 10.0),
    }

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train with early stopping
    watchlist = [(dtrain, "train"), (dval, "eval")]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=20, verbose_eval=False)
    
    # Predict on validation set
    y_pred = model.predict(dval)
    
    # Calculate MSE
    mse = mean_squared_error(y_val, y_pred)
    return mse


# Starting Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost on genotype data for phenotype prediction")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--result_file', type=str, required=True, help='Path to save the result output file')
    args = parser.parse_args()

    result_file = args.result_file
    path_training_dataset = args.data_path

    # result file
    with open(result_file, "w") as f:
        f.write("XGBoost Results\n")
        f.write("=" * 40 + "\n")

    # Validation and Test set
    X_val = pd.read_parquet(path_training_dataset + "val.parquet")
    y_val = pd.read_parquet(path_training_dataset + "val_pheno.parquet")
    X_test = pd.read_parquet(path_training_dataset + "test.parquet")
    y_test = pd.read_parquet(path_training_dataset + "test_pheno.parquet")
    # Convert datasets into DMatrix format for XGBoost
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train on Real Dataset
    print(f"Training on real dataset")
    X_train = pd.read_parquet(path_training_dataset + "real.parquet")
    y_train = pd.read_parquet(path_training_dataset + "pheno.parquet")
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Run Optuna optimization for this dataset
    study = optuna.create_study(direction="minimize")  # Minimize RMSE
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    # Get best parameters for this dataset
    best_params = study.best_params
    print(f"Best Hyperparameters for dataset: {best_params}")
    # Train final model using best parameters and early stopping
    best_model = xgb.train(best_params, dtrain, num_boost_round=1000, evals=[(dval, 'eval')], early_stopping_rounds=20, verbose_eval=True)
    # Predict on test set
    y_pred_test = best_model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred_test)
    corr = pearsonr(y_test.squeeze(), y_pred_test)[0]
    
    with open(result_file, "a") as f:
        f.write(f"Dataset Real:\n")
        f.write(f"Test MSE: {mse:.4f}\n")
        f.write(f"Pearson Correlation: {corr:.4f}\n")
        f.write("-" * 30 + "\n")
    print(f"Dataset Real - Test MSE: {mse:.4f}, Pearson Correlation: {corr:.4f}")

    # Train on Synthetic Dataset
    print(f"Training on synthetic dataset")
    X_train = pd.read_parquet(path_training_dataset + "syn.parquet")
    y_train = pd.read_parquet(path_training_dataset + "pheno.parquet")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Run Optuna optimization for this dataset
    study = optuna.create_study(direction="minimize")  # Minimize RMSE
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    # Get best parameters for this dataset
    best_params = study.best_params
    print(f"Best Hyperparameters for dataset: {best_params}")
    # Train final model using best parameters and early stopping
    best_model = xgb.train(best_params, dtrain, num_boost_round=1000, evals=[(dval, 'eval')], early_stopping_rounds=20, verbose_eval=True)
    # Predict on test set
    y_pred_test = best_model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred_test)
    corr = pearsonr(y_test.squeeze(), y_pred_test)[0]
    
    with open(result_file, "a") as f:
        f.write(f"Dataset Syn:\n")
        f.write(f"Test MSE: {mse:.4f}\n")
        f.write(f"Pearson Correlation: {corr:.4f}\n")
        f.write("-" * 30 + "\n")
    print(f"Dataset Syn - Test MSE: {mse:.4f}, Pearson Correlation: {corr:.4f}")