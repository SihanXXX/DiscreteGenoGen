You can train the model using the command below:

```bash
python train.py --data_path "/path/to/data/"
```

, where data_path directory must contain:

- `train.parquet`: Training genotype dataset (the PCA latent representation)
- `pheno.parquet`: Phenotype labels corresponding to the training set
- `val.parquet`: Validation genotype dataset (the original representation)
- `val_pheno.parquet`: Phenotype labels corresponding to the validation set

The pretrained model is too large to upload to GitHub, so we've made it available on [Zenodo](https://zenodo.org/records/16571171):
[Click here to download the model](https://zenodo.org/record/16571171/files/dm_cow_all.pth?download=1)