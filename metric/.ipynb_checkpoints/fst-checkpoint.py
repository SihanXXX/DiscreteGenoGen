import numpy as np
import pandas as pd

# The Fixation Index is a widely used statistical measure in population genetics to quantify genetic differentiation between two or more populations. 

def aggregated_fst(df1, df2):
    """
    Calculate aggregated fixtion index for multiple SNPs between two datasets.

    Args:
        df1 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.
        df2 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.

    Returns:
        F_ST_aggregated (float): aggregated fixation index. Fixation index F_st for a SNP i is given by F_st(i) = (expected heterozygosity in the combined population - average expected heterozygosity within each subpopulation)/expected heterozygosity in the combined population. The aggregated fixation index is a weighted average of fixation index for all the SNPs, given by weighted_F_st = sum(expected heterozygosity in the combined population for SNP i * F_st(i))/sum(expected heterozygosity in the combined population for SNP i)
    """
    df1_arr, df2_arr = df1.to_numpy(), df2.to_numpy()
    
    # Vectorized computation of allele frequencies
    al_freq_1 = (2 * (df1_arr == 2).sum(axis=0) + (df1_arr == 1).sum(axis=0)) / (2 * df1_arr.shape[0])
    al_freq_2 = (2 * (df2_arr == 2).sum(axis=0) + (df2_arr == 1).sum(axis=0)) / (2 * df2_arr.shape[0])

    # Global frequency and heterozygosities
    p_total = (al_freq_1 + al_freq_2) / 2
    H_S = (1 - (al_freq_1**2 + (1 - al_freq_1)**2) + 1 - (al_freq_2**2 + (1 - al_freq_2)**2)) / 2
    H_T = 1 - (p_total**2 + (1 - p_total)**2)

    # Vectorized calculation of F_ST
    valid_ht = H_T > 0
    F_ST_per_snp = np.zeros_like(H_T)
    F_ST_per_snp[valid_ht] = (H_T[valid_ht] - H_S[valid_ht]) / H_T[valid_ht]
    F_ST_per_snp[F_ST_per_snp < 0] = 0

    # Aggregated F_ST using weighted sum
    F_ST_aggregated = np.sum(H_T * F_ST_per_snp) / np.sum(H_T)

    return F_ST_aggregated
