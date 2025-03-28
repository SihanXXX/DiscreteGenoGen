import matplotlib.pyplot as plt

def compute_homozygosity_rate(df, allel = 'alternative'):
    """
    Compute homozygosity rate for each SNP.
    
    Parameters:
        df (pd.DataFrame): genotype matrix (individuals x SNPs).
        allel (string): choose between 'reference' or 'alternative'
        
    Returns:
        pd.Series: Homozygosity rate for each SNP.
    """
    if allel == "alternative":
        return (df == 2).mean(axis=0)
    else:
        return (df == 0).mean(axis=0) 

def compute_heterozygosity_rate(df):
    """
    Compute heterozygosity rate for each SNP.
    
    Parameters:
        df (pd.DataFrame): genotype matrix (individuals x SNPs).
        
    Returns:
        pd.Series: Heterozygosity rate for each SNP.
    """
    return (df == 1).mean(axis=0)

def plot_geno_freq(df1, df2, xlabel, ylabel, save_path_img):
    """
    Plot a scatter plot to compare geno frequencies between two datasets.

    Args:
        df1 (pd.DataFrame): genotype matrix (individuals x SNPs).
        df2 (pd.DataFrame): genotype matrix (individuals x SNPs).
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        save_path_img (str): Path to save the resulting scatter plot as an image.

    Returns:
        None. Displays the scatter plot and saves it to the specified path.
    """

    # Calculate homozygosity rates of alternative allels for both datasets
    df1_homo_alter = compute_homozygosity_rate(df1)
    df2_homo_alter = compute_homozygosity_rate(df2)

    # Calculate homozygosity rates of reference allels for both datasets
    df1_homo_ref = compute_homozygosity_rate(df1, "reference")
    df2_homo_ref = compute_homozygosity_rate(df2, "reference")

    # Calculate heterozygosity rates for both datasets
    df1_het = compute_heterozygosity_rate(df1)
    df2_het = compute_heterozygosity_rate(df2)

    # Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(df1_het, df2_het, alpha=0.5, color="#008C90", label="Heterozygosity Rate")
    plt.scatter(df1_homo_alter, df2_homo_alter, alpha=0.5, color="#800080", label="Homozygosity Rate for alternative allel")
    plt.scatter(df1_homo_ref, df2_homo_ref, alpha=0.5, color="#FF4500", label="Homozygosity Rate for reference allel")
    plt.plot([0, 1], [0, 1], color="#FF6F43", linestyle='--', label='y = x (Ideal Match)')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.savefig(save_path_img, format="png")
    plt.savefig(save_path_img, format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    





    
