import umap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def geno_UMAP(df1, df2, label1, label2, save_path):
    """UMAP visualization between two genotype dataframes
    ----
    Parameters:
        df1 (dataframe): first genotype dataframe
        df2 (dataframe): second genotype dataframe
        label1 (string): label of the df1 to be displayed in the 2d UMAP plot
        label2 (string): label of the df1 to be displayed in the 2d UMAP plot
        save_path (string): the path where the umap plot will be saved
    """
    nb_df1 = df1.shape[0]
    combined_df1_df2 = pd.concat([
        df1.assign(Origin=label1),
        df2.assign(Origin=label2)
    ], ignore_index=True)
    reducer = umap.UMAP(random_state=42)
    snps_umap = reducer.fit_transform(combined_df1_df2.drop(combined_df1_df2.columns[-1], axis=1))
    snps_1 = snps_umap[:nb_df1, :]
    snps_2 = snps_umap[nb_df1:, :]

    # Plot UMAP results
    plt.figure(figsize=(10, 7))
    # Plot df1
    plt.scatter(snps_1[:, 0], snps_1[:, 1], alpha=0.6, label=label1, color="#832a78", s=50)
    # Plot df2
    plt.scatter(snps_2[:, 0], snps_2[:, 1], alpha=0.5, label=label2, color="#87bccb", s=50)
    plt.xlabel('Component 1', fontsize=12, color="#000000")
    plt.ylabel('Component 2', fontsize=12, color="#000000")
    plt.legend()
    # plt.savefig(save_path, format="png")
    plt.savefig(save_path, format='eps', dpi=600, bbox_inches='tight')
    plt.show()