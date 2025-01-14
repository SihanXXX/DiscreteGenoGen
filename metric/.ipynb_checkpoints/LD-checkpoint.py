import allel
from scipy.spatial.distance import squareform
from scipy.stats import binned_statistic
from scipy.stats import sem
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def LD(df):
    """
    Calculate LD for each pair of SNPs in the dataframe. LD can be understood as a measure of statistical association between loci. However, when working with diploid genotype data, the gametic phase (the arrangement of alleles on homologous chromosomes) is often unknown, making it challenging to directly determine haplotypes. Here we apply a method designed for LD estimation under conditions of phase ambiguity.
   (Reference: Rogers, A. R., & Huff, C. (2009). Linkage disequilibrium between loci with unknown phase. Genetics, 182(3), 839–844. https://doi.org/10.1534/genetics.108.093153)
    
    Parameters:
        df (pd.DataFrame): genotype matrix (individuals x SNPs).
        
    Returns:
        r (ndarray,float): length of (nb_SNPs * (nb_SNPs - 1) / 2
        r2 (ndarray,float): length of (nb_SNPs * (nb_SNPs - 1) / 2, the upper-diagonal elements in the pairwise LD matrix
        r2_matrix (ndarray,float): shape of (nb_SNPs * nb_SNPs), the pairwise LD matrix
    """
    r = allel.rogers_huff_r(df.values.T)

    return r, r ** 2, squareform(r ** 2)
    

def plot_LD(df1, df2, save_path_img):
    """
    Generate a heatmap to compare Linkage Disequilibrium (LD) between two datasets.
    The lower triangular part of the heatmap represents the LD values from df1, 
    while the upper triangular part represents the LD values from df2.

    Args:
        df1 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.
        df2 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.
        save_path_img (str): File path to save the generated heatmap as an image.

    Returns:
        None: Displays the heatmap and saves it to the specified path.
    """
    # Ensure the input dataframes have the same shape
    if df1.shape != df2.shape:
        raise ValueError("Both genotype matrices must have the same shape.")

    # Compute pairwise LD matrices using the provided LD function
    _,_,ld_mat1 = LD(df1)
    _,_,ld_mat2 = LD(df2)

    # Initialize the combined LD matrix with zeros
    combined_LD = np.zeros_like(ld_mat1)

    # Fill lower triangle with ld_mat1 and upper triangle with ld_mat2
    for i in range(ld_mat1.shape[0]):
        for j in range(ld_mat1.shape[1]):
            if i >= j:  # Lower triangle for df1
                combined_LD[i, j] = ld_mat1[i, j]
            else:       # Upper triangle for df2
                combined_LD[i, j] = ld_mat2[i, j]

    # Plot the LD heatmap
    custom_cmap = sns.light_palette("#008C90", as_cmap=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        combined_LD,
        annot=False,  # Set True to see correlation values
        cmap=custom_cmap,
        vmin=0, vmax=1,
        cbar_kws={'label': 'LD'})
    
    # plt.title("LD Heatmap: Real (below diagonal) vs Fake (above diagonal)")
    plt.savefig(save_path_img, format="png")
    plt.show()


def plot_LD_decay(df1, df2, positions, distance_threshold, min_dist, max_dist, label1, label2, save_path_img):
    """
    Compare Linkage Disequilibrium (LD) Decay with respect to physical distance for two datasets.
    
    Args:
        df1 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.
        df2 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.
        positions (numpy.ndarray): Array containing the physical positions of SNPs in a chromosome.
                                   (The number of cols in df1, df2, and length of positions must match).
        distance_threshold (int): Maximum physical distance for SNP pairs to be considered.
        min_dist (int): Minimum physical distance to be plotted.
        max_disy (int): Maximum physical distance to be plotted.
        label1 (str): Label for df1.
        label2 (str): Label for df2.
        save_path_img (str): File path to save the plot.

    Returns:
        None: Displays the plot and saves it to the specified path.
    """
     # Validate inputs
    if len(positions) != df1.shape[1] or len(positions) != df2.shape[1]:
        raise ValueError("The number of positions must match the number of SNPs in both datasets.")
    
    # Compute physical distances
    distances = calculate_pairwise_phy_distance(positions)

    # Compute LD
    _, ld_df1, _ = LD(df1)
    _, ld_df2, _ = LD(df2)

    # Apply the distance threshold filter         
    filtered_distances = distances[distances <= distance_threshold]
    filtered_ld_1 = ld_df1[distances <= distance_threshold]
    filtered_ld_2 = ld_df2[distances <= distance_threshold]

    # Plot the scatter plot
    plt.figure(figsize=(10, 8))

    # Create logarithmic bins for better visualization
    bins = np.logspace(np.log10(min_dist), np.log10(max_dist), 50)

    # Plot for df1
    bin_means_1, bin_edges_1, _ = binned_statistic(filtered_distances, filtered_ld_1, statistic='mean', bins=bins)
    # Scatter plot
    plt.scatter(filtered_distances, filtered_ld_1, alpha=0.2, color='#007F85', label=label1, s=10)
    # Add a line for binned means
    bin_centers_1 = 0.5 * (bin_edges_1[:-1] + bin_edges_1[1:])  # Calculate bin centers
    plt.plot(bin_centers_1, bin_means_1, color='#FF6F43', linewidth=2, label='Mean LD for '+label1)
    # Calculate standard error for each bin
    bin_sems_1, _, _ = binned_statistic(filtered_distances, filtered_ld_1, statistic=sem, bins=bins)
    plt.fill_between(bin_centers_1, bin_means_1 - bin_sems_1, bin_means_1 + bin_sems_1, color='#FF6F43', alpha=0.3)

    # Plot for df2
    bin_means_2, bin_edges_2, _ = binned_statistic(filtered_distances, filtered_ld_2, statistic='mean', bins=bins)
    # Scatter plot 
    plt.scatter(filtered_distances, filtered_ld_2, alpha=0.2, color='#8A2BE2', label=label2, s=10)
    # Add a line for binned means
    bin_centers_2 = 0.5 * (bin_edges_2[:-1] + bin_edges_2[1:])  # Calculate bin centers
    plt.plot(bin_centers_2, bin_means_2, color='#FFC20A', linewidth=2, label='Mean LD for '+label2)
    # Calculate standard error for each bin
    bin_sems_2, _, _ = binned_statistic(filtered_distances, filtered_ld_2, statistic=sem, bins=bins)
    plt.fill_between(bin_centers_2, bin_means_2 - bin_sems_2, bin_means_2 + bin_sems_2, color='#FF6F43', alpha=0.3)

    # Beautify the plot
    plt.xscale('log')  # Logarithmic scale for distances
    plt.xlim(min_dist, max_dist)  # Focus only on the specified range
    plt.ylim(0, 1)  # LD values range between 0 and 1
    plt.xlabel('Physical Distance (base pairs)', fontsize=12)
    plt.ylabel('Linkage Disequilibrium (LD)', fontsize=12)
    #plt.title('LD Decay Analysis', fontsize=14)
    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path_img, format="png")
    plt.show()


def calculate_pairwise_phy_distance(positions):
    """
    Compute pairwise physical distance for all the SNPs, measured by base-pair

    Parameters:
        positions (numpy.ndarray): An array which contains position of all the SNPs in a chromosome

    Returns:
        distances (numpy.ndarray): Distances between SNP pairs.
    """
    n_snps = positions.shape[0]
    distances = []

    for i in range(n_snps):
        for j in range(i + 1, n_snps):
            distance = abs(positions[j] - positions[i])
            distances.append(distance)

    return np.array(distances)