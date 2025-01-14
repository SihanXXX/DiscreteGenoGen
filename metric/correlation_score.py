"""
The idea of correlation score is to compute a Gamma correlation coefficient that compares the correlation structures of two genotype matrices

Improved from original code:
-----
https://github.com/rvinas/adversarial-gene-expression.git

Reference:
-----
Ramon Viñas, Helena Andrés-Terré, Pietro Liò, Kevin Bryson,
Adversarial generation of gene expression data,
Bioinformatics, Volume 38, Issue 3, February 2022, Pages 730–737,
https://doi.org/10.1093/bioinformatics/btab035
"""

import numpy as np

def pearson_correlation(x: np.array, y: np.array):
    """
    Computes similarity measure between each pair of snps in the bipartite graph x <-> y
    ----
    Parameters:
        x (np.array): SNP matrix 1. Shape=(nb_samples, nb_SNPs_1)
        y (np.array): SNP matrix 2. Shape=(nb_samples, nb_SNPs_2)
    Returns:
        Matrix with shape (nb_SNPs_1, nb_SNPs_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        S = (a - a_off) / a_std
        S[np.isnan(S)] = (a - a_off)[np.isnan(S)]
        return S

    assert x.shape[0] == y.shape[0]
    x_ = standardize(x)
    y_ = standardize(y)
    return np.dot(x_.T, y_) / x.shape[0]


def upper_diag_list(m_: np.array):
    """
    Returns the condensed list of all values in the upper-diagonal of m_
    ----
    Parameters:
        m_ (np.array): array of float. Shape = (N, N)
    Returns:
        np.array: 1D array of values in the upper-diagonal of m_.
                  Shape = (N*(N-1)/2,)
    """
    return m_[np.triu_indices_from(m_, k=1)]


def correlations_list(x: np.array, y: np.array):
    """
    Generates correlation list between all pairs of SNPs in the bipartite graph x <-> y
    ----
    Parameters:
        x (np.array): SNPs matrix 1. Shape=(nb_samples, nb_SNPs_1)
        y (np.array): SNPs matrix 2. Shape=(nb_samples, nb_SNPs_2)
    Returns:
        corr_list (np.array): 1D array of values in the upper-diagonal of x and y's correlation matrix
    """
    corr_list = upper_diag_list(pearson_correlation(x, y))

    return corr_list


def corr_score(x: np.array, y: np.array):
    """
    Compute correlation score for two given genotype matrices
    ----
    Parameters:
        x (np.array): SNPs matrix. Shape=(nb_samples_1, nb_SNPs)
        y (np.array): SNPs matrix. Shape=(nb_samples_2, nb_SNPs)
    Returns:
        gamma_dx_dy (float): correlation coefficient score
    """
    dists_x = 1 - correlations_list(x, x)
    dists_y = 1 - correlations_list(y, y)
    gamma_dx_dy = pearson_correlation(dists_x, dists_y)

    return gamma_dx_dy