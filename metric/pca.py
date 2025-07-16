import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import ot

def geno_PCA (df1, df2, label1, label2, save_path):
    """PCA visualization between two genotype dataframes
    ----
    Parameters:
        df1 (dataframe): first genotype dataframe
        df2 (dataframe): second genotype dataframe
        label1 (string): label of the df1 to be displayed in the 2d PCA plot
        label2 (string): label of the df2 to be displayed in the 2d PCA plot
        save_path (string): the path where the pca plot will be saved
    """
    nb_df1 = df1.shape[0]
    combined_df1_df2 = pd.concat([
        df1.assign(Origin=label1),
        df2.assign(Origin=label2)
    ], ignore_index=True)

    pca = PCA(n_components=2)
    snps_2d = pca.fit_transform(combined_df1_df2.drop(combined_df1_df2.columns[-1], axis=1))
    snps_2d_1 = snps_2d[:nb_df1, :]
    snps_2d_2 = snps_2d[nb_df1:, :]

    # Plot PCA results
    plt.figure(figsize=(10, 7))
    # Plot df1
    plt.scatter(snps_2d_1[:, 0], snps_2d_1[:, 1], alpha=0.6, label=label1, color="#832a78", s=50)
    # Plot df2
    plt.scatter(snps_2d_2[:, 0], snps_2d_2[:, 1], alpha=0.5, label=label2, color="#87bccb", s=50)
    plt.xlabel('PC1', fontsize=12, color="#000000")
    plt.ylabel('PC2', fontsize=12, color="#000000")
    plt.legend()
    # plt.savefig(save_path, format="png")
    plt.savefig(save_path+".eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(save_path+".pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()


def cumu_var (df1, df2, label1, label2, nb_pc, save_path):
    """Cummulative variance comparaison plot for two genotype dataframes
    ----
    Parameters:
        df1 (dataframe): first genotype dataframe
        df2 (dataframe): second genotype dataframe
        label1 (string): label of the df1 to be displayed in the cumulative variance plot
        label2 (string): label of the df1 to be displayed in the cumulative variance plot
        nb_pc (integer): the number of the principal components to take into account in PCA
        save_path (string): the path where the plot will be saved
    Return 
        wasserstein_dis (float): wasserstein distance between the nb_pc PCs of two dataframes
    """
    pca_df1 = PCA(n_components=nb_pc)
    pca_df2 = PCA(n_components=nb_pc)
    pca_df1_result = pca_df1.fit_transform(df1)
    pca_df2_result = pca_df2.fit_transform(df2)

    # Compute cumulative explained variance
    cumu_var_df1 = np.cumsum(pca_df1.explained_variance_ratio_)
    cumu_var_df2 = np.cumsum(pca_df2.explained_variance_ratio_)

    # Find the number of components to reach 80% variance
    threshold = 80
    num_pc_df1 = np.argmax((cumu_var_df1*100) >= threshold) + 1
    num_pc_df2 = np.argmax((cumu_var_df2*100) >= threshold) + 1

    # Plot cumulative explained variance
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, len(cumu_var_df1) + 1), cumu_var_df1, marker='o', color = "#fddf21", label=label1)
    plt.plot(range(1, len(cumu_var_df2) + 1), cumu_var_df2, marker='x', color = "#6fa84e", label=label2)
    plt.axvline(x=num_pc_df1, color='#fddf21', linestyle='--', label=f'{num_pc_df1} PCs to reach 80% variance')
    plt.axvline(x=num_pc_df2, color='#6fa84e', linestyle='--', label=f'{num_pc_df2} PCs to reach 80% variance')
    plt.xlabel('Number of PCs', fontsize=12, color="#000000")
    plt.ylabel('Cumulative Explained Variance', fontsize=12, color="#000000")
    #plt.title('Cumulative Explained Variance by Number of Principal Components')
    plt.legend() 
    plt.grid()
    # plt.savefig(save_path, format="png")
    plt.savefig(save_path+".eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(save_path+".pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

    return wasserstein_distance(pca_df1_result, pca_df2_result)


def wasserstein_distance (df1, df2):
    """
    Parameters:
        df1 (dataframe): first genotype dataframe
        df2 (dataframe): second genotype dataframe
    Return 
        wasserstein_dis (float): wasserstein distance between two dataframes
    """        
    cost_matrix = ot.dist(df1, df2)

    # Uniform weights for both datasets
    weights_df1 = np.ones((df1.shape[0],)) / df1.shape[0]
    weights_df2 = np.ones((df2.shape[0],)) / df2.shape[0]
    # Compute Wasserstein distance using POT
    wasserstein_distance = ot.emd2(weights_df1, weights_df2, cost_matrix)

    return wasserstein_distance
    

def geno_PCA_32PC (df1, df2, label1, label2, save_path):
    """
    Perform PCA on the combined dataset and plot PC1 vs PC2, PC3 vs PC4, ..., PC31 vs PC32.
    
    ----
    Parameters:
        df1 (dataframe): first genotype dataframe
        df2 (dataframe): second genotype dataframe
        label1 (string): label of the df1 to be displayed in the 2d PCA plot
        label2 (string): label of the df2 to be displayed in the 2d PCA plot
        save_path (string): the path where the pca plot will be saved
        
    Returns:
        None: Displays the 4x4 grid of PCA plots.
    """
    # Combine real and artificial data for PCA
    combined_data = np.vstack([df1.to_numpy(), df2.to_numpy()])
    pca = PCA(n_components=32)
    combined_pca = pca.fit_transform(combined_data)

    # Split the PCA-transformed data back into df1 and df2
    n_df1 = df1.shape[0]
    df1_pca = combined_pca[:n_df1, :]
    df2_pca = combined_pca[n_df1:, :]

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.sum(explained_variance)
    print(f"Cumulative variance explained by the first 32 PCs: {cumulative_variance:.4f} ({cumulative_variance*100:.2f}%)")

    # Create a 4x4 grid for plotting
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # Plot PC1 vs PC2, PC3 vs PC4, ..., PC31 vs PC32
    pc_idx = 0
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]

            # Select the current pair of PCs
            pc_x = pc_idx
            pc_y = pc_idx + 1

            # Scatter plot for real and artificial data
            ax.scatter(df1_pca[:, pc_x], df1_pca[:, pc_y], alpha=0.3, label=label1, color="#008C90", s=10)
            ax.scatter(df2_pca[:, pc_x], df2_pca[:, pc_y], alpha=0.3, label=label2, color="#D0FF00", s=10)

            # Titles and labels
            ax.set_title(f"PC{pc_x + 1} vs PC{pc_y + 1}")
            ax.set_xlabel(f"PC{pc_x + 1}")
            ax.set_ylabel(f"PC{pc_y + 1}")

            pc_idx += 2 

            # Add label on the first plot
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    # plt.savefig(save_path, format="png")
    plt.savefig(save_path+".eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(save_path+".pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()
    