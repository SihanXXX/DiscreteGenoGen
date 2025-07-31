You can train the model using the command below:

```bash
python train.py --data_path "/path/to/data/"
```

, where data_path directory must contain:

- `train.parquet`: Training genotype dataset
- `pheno.parquet`: Phenotype labels corresponding to the training set
- `val.parquet`: Validation genotype dataset
- `val_pheno.parquet`: Phenotype labels corresponding to the validation set

The pretrained model is too large to upload to GitHub, so we've made it available on [Zenodo](https://zenodo.org/records/16411670):
[Click here to download the model](https://zenodo.org/record/16411670/files/wgan_cow_all.pth.xz?download=1)