# Deep Generative Models for Synthetic Discrete Genotype Simulation

This repository accompanies the paper:  
**_Deep Generative Models for Synthetic Discrete Genotype Simulation_**  
by **Sihan Xie**

## 🔍 Overview

This project implements several deep generative models—including Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Wasserstein GANs (WGANs), and Diffusion Models—to simulate synthetic genotype data. We propose model adaptations specifically tailored to the discrete nature of genotype representation and evaluate them across multiple metrics rooted in both quantitative genetics and deep learning.

![Project Schema](./GenSNP_schema.png)

## 🧬 Datasets

- **Cow Cohort**  
  - 93484 Holstein cows  
  - 50161 SNPs across 29 pairs of autosomal chromosomes  
  - Phenotype: Fat content (FC), represented by Yield Deviation (YD)

- **Human Cohort**  
  - Subsets derived from the [UK Biobank](https://www.ukbiobank.ac.uk/)  
  - Used to test the scalability and generalizability of generative models across species and population groups 
  - Phenotypes: Height and Sex

## 📦 Repository Structure

```bash
.
├── demo/                   # Reproducible demo notebooks to replicate our experiments result (recommended starting point)
├── metric/                 # Implementation of evaluation metrics for synthetic genotype data
├── metric_analysis_result/ # Analysis results of metric behavior and robustness
├── model/                  # Model architectures and training code
├── pca/                    # PCA results used as latent representation for diffusion models
├── pheno_prediction/       # Predictive Models for Genotype-to-Phenotype Prediction
├── GenSNP_schema.png      
└── README.md               
