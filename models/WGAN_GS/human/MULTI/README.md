You can train the model using the command below:

```bash
python train.py --data_path "/path/to/data/"
```

, where data_path directory must contain:

- `train.parquet`: Training genotype dataset
- `pheno.parquet`: Phenotype labels corresponding to the training set
- `val.parquet`: Validation genotype dataset
- `val_pheno.parquet`: Phenotype labels corresponding to the validation set

The `pheno.parquet` and `val_pheno.parquet` files should contain two columns:

- Sex: Encoded as 1 for male and 2 for female
- Height: Individual’s height in centimetres (cm)