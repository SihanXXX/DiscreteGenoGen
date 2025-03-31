import matplotlib.pyplot as plt

def allele_freq(df):
    """
    Calculate the minor allele frequencies (MAF) for each SNP in the dataset.

    Args:
        df (pd.DataFrame): A DataFrame where each column represents an SNP, and
                           each row represents an individual's genotype. 
                           Genotypes are encoded as:
                           - 0: Homozygous for reference allele
                           - 1: Heterozygous
                           - 2: Homozygous for minor allele

    Returns:
        pd.Series: A Series where the index is SNP column names and the values are
                   the minor allele frequencies (MAF) for each SNP.
    """
    # Count the number of minor alleles per SNP using vectorized operations
    total_minor_alleles = (df == 1).sum(axis=0) + 2 * (df == 2).sum(axis=0)

    # Total number of alleles per SNP (2 alleles per individual)
    total_alleles = 2 * len(df)

    # Calculate minor allele frequencies (vectorized)
    frequencies = total_minor_alleles / total_alleles

    return frequencies


def plot_allele_freq(df1, df2, xlabel, ylabel, save_path_img):
    """
    Plot a scatter plot to compare allele frequencies between two datasets.

    Args:
        df1 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.
        df2 (pd.DataFrame): Genotype matrix where rows represent individuals and columns represent SNPs.
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        save_path_img (str): Path to save the resulting scatter plot as an image.

    Returns:
        None. Displays the scatter plot and saves it to the specified path.
    """
    plt.figure(figsize=(10, 8))
    
    # Compute MAF
    maf1 = allele_freq(df1)
    maf2 = allele_freq(df2)
    # Scatter plot for the SNP
    plt.scatter(maf1.tolist(), maf2.tolist(), color="#008C90", alpha=0.8)

    # Add a reference line (y = x) to show ideal agreement
    plt.plot([0, 1], [0, 1], 'r--', color="#FF6F43", label='y = x (Ideal Match)')
    
    # Add title, labels, legend, and grid
    # plt.title('Allele Frequency Comparison', fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    
    # Set axis limits for allele frequency range
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Save the plot to the specified file path
    # plt.savefig(save_path_img, format="png")
    plt.savefig(save_path_img+".eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(save_path_img+".pdf", format='pdf', dpi=600, bbox_inches='tight')
    
    # Display the plot
    plt.show()