# Phenotype Prediction from Genotype Data

This repository contains simple implementations of **XGBoost** and **MLP (Multi-Layer Perceptron)** models to predict **phenotypes** from **genotype** data. The goal is to evaluate how well synthetic genotype data can achieve comparable performance as real genotype data. The repository leverages the **Optuna** library for hyperparameter optimization.

## Repository Structure

- `cow_geno_pheno_pred_xgb.py`: Implementation of the **XGBoost** model for phenotype prediction.
- `cow_geno_pheno_pred_mlp.py`: Implementation of the **MLP (Multi-Layer Perceptron)** model for phenotype prediction.

To run the `cow_geno_pheno_pred_xgb.py` script:

```bash
python cow_geno_pheno_pred_xgb.py \
  --data_path "path/to/data_folder/" \
  --result_file "path/to/result_file.txt"
```

To run the `cow_geno_pheno_pred_mlp.py` script:

```bash
python cow_geno_pheno_pred_mlp.py \
  --data_path "path/to/data_folder/" \
  --result_file "path/to/result_file.txt"
```

, where in the "data_path", you should have real.parquet(real genotype training dataset), syn.parquet(synthetic genotype training dataset), pheno.parquet(phenotype training dataset), val.parquet(genotype validation dataset), val_pheno.parquet(phenotype validation dataset), test.parquet(genotype test dataset), test_pheno.parquet(phenotype test dataset).
