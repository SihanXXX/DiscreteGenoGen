You can train the model using the command below:

```bash
python train.py --data_path "/path/to/data/"
```

, where data_path directory must contain:

- `train.parquet`: Training genotype ataset
- `pheno.parquet`: Phenotype labels corresponding to the training set
- `val.parquet`: Validation genotype dataset
- `val_pheno.parquet`: Phenotype labels corresponding to the validation set

The pretrained model is too large to upload to GitHub, so we've made it available on [Zenodo](https://zenodo.org/records/16411670?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA0YzJkNDM4LTIzZDYtNDk1ZC04NmU5LWYwM2E1ZTZmMWFiMSIsImRhdGEiOnt9LCJyYW5kb20iOiI5ODI4ODZlOTdiZDA0NGY1YjZjOTBiODZhYzQxYTAyZCJ9.NHClV70SP6bwrkacESudh6FrqeV-yUQ6-6YEgojn_vUvVOeBO2LDuoc0CowRfoIpdX8ju1muFFmdreBALvHLOA):
[Click here to download the model](https://zenodo.org/record/16411670/files/wgan_cow_all.pth.xz?download=1)