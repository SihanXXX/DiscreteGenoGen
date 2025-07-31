You can train the model using the command below:

```bash
python train.py --data_path "/path/to/data/"
```

, where data_path directory must contain:

- `train.parquet`: Training genotype dataset (the PCA latent representation)
- `pheno.parquet`: Phenotype labels corresponding to the training set
- `val.parquet`: Validation genotype dataset (the original representation)
- `val_pheno.parquet`: Phenotype labels corresponding to the validation set

The `pheno.parquet` and `val_pheno.parquet` files should contain two columns:

- Sex: Encoded as 1 for male and 2 for female
- Height: Individual’s height in centimetres (cm)

And you also need to have `pca_components` and `pca_mean` saved in [pca](../../../../pca) repository to reconstruct from the latent representation.