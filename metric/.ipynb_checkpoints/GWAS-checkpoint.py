import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Genome Wide Association Studies
def gwas(geno_df, pheno_array):
    """
    Perform GWAS using marginal linear regression (we regress the phenotype only on 1 snp at each time).

    Args:
        geno_df (pd.DataFrame): SNP genotype matrix
        pheno_array (np.ndarray): Array of phenotype trait values

    Returns:
        pd.DataFrame: GWAS results containing SNP names, effect sizes (Beta), and p-values
    """

    # Ensure inputs are consistent
    if len(pheno_array) != geno_df.shape[0]:
        raise ValueError("The number of individuals in the phenotype array must match the SNP matrix.")

    # Prepare results storage
    results = []

    # Loop through each SNP column and run linear regression
    for snp in geno_df.columns:
        snp_data = geno_df[snp].values

        # Check if the SNP column is constant (no variation)
        if np.all(snp_data == snp_data[0]):
            # Assign non-significant p-value when constant
            results.append({"SNP": snp, "Beta": 0, "P-value": 1.0})  # Non-significant p-value
            continue

        # Adding intercept for regression
        X = sm.add_constant(snp_data)  # [1, SNP value] for regression
        y = pheno_array

        # Linear regression with statsmodels
        model = sm.OLS(y, X).fit()

        # Extract effect size and p-value
        try:
            beta = model.params[1]  # SNP effect size
            p_value = model.pvalues[1]  # P-value for SNP association
            
        except IndexError:
            beta = 0
            p_value = 1.0  # Assign non-significant p-value

        # Store results
        results.append({"SNP": snp, "Beta": beta, "P-value": p_value})

    return pd.DataFrame(results)


def plot_gwas(geno_df, pheno_array, chrom_dict, save_path_img):
    """
    Create a Manhattan plot for GWAS results.
    Args:
        geno_df (pd.DataFrame): SNP genotype matrix
        pheno_array (np.ndarray): Array of phenotype trait values
        chrom_dict (dict): Dictionary where keys are chromosome numbers and values are indices of the last SNP belonging to the chromosome, the starting indice is 0
        save_path_img (str): File path to save the generated image
    Returns:
        None: Displays the Manhattan plot
    """
    # Perform GWAS
    gwas_results = gwas(geno_df, pheno_array)
    p_values = gwas_results['P-value']
    log_p_values = -np.log10(p_values)

    chrom_colors = ['#17becf', '#9edae5']
    x_ticks = []  # To store the middle point of each chromosome for labeling
    x_labels = []  # Chromosome labels
    current_position = 0  # Track cumulative position of SNPs

    # Manhattan Plot
    plt.figure(figsize=(12, 6))

    for chrom, last_index in chrom_dict.items():
        start_index = current_position
        end_index = last_index
        chrom_range = np.arange(start_index, end_index)

        # Plot SNPs for the current chromosome
        plt.scatter(
            chrom_range,
            log_p_values[start_index:end_index],
            color=chrom_colors[chrom % 2],
            s=10
        )
        
        # Update cumulative position
        x_ticks.append((start_index + end_index) // 2)  # Middle of the chromosome
        x_labels.append(f"Chr{chrom}")
        current_position = end_index

    # Add significance threshold line
    threshold = -np.log10(0.05 / len(p_values))  # Bonferroni correction
    plt.axhline(y=threshold, color='#e377c2', linestyle='--', linewidth=0.8, label="Bonferroni Threshold")

    # Beautify the plot
    plt.xticks(x_ticks, x_labels, rotation=90)
    plt.xlabel("Chromosomes", fontsize=12)
    plt.ylabel("-log10(P-value)", fontsize=12)
    # plt.title("Manhattan Plot of GWAS Results", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    # plt.savefig(save_path_img, format="png")
    plt.savefig(save_path_img, format='eps', dpi=600, bbox_inches='tight')
    plt.show()


def plot_compare_gwas(geno_df_1, pheno_array_1, geno_df_2, pheno_array_2, chrom_dict, label1, label2, save_path_img):
    """
    Compare two GWAS results visually using a symmetrical Manhattan plot.
    
    Args:
        geno_df_1 (pd.DataFrame): SNP genotype matrix
        pheno_array_1 (np.ndarray): Array of phenotype trait values
        geno_df_2 (pd.DataFrame): SNP genotype matrix
        pheno_array_2 (np.ndarray): Array of phenotype trait values
        chrom_dict (dict): Dictionary where keys are chromosome numbers and values are indices of the last SNP belonging to the chromosome, the starting indice is 0
        label1 (str): Label for geno_df_1 
        label2 (str): Label for geno_df_2 
        save_path_img (str): File path to save the generated image
    
    Returns:
        None: Displays the symmetrical Manhattan plot for comparison.
    """
    
    # Perform GWAS
    gwas_results_1 = gwas(geno_df_1, pheno_array_1)
    p_values_1 = gwas_results_1['P-value']
    beta_1 = gwas_results_1['Beta']
    log_p_values_1 = -np.log10(p_values_1)
    gwas_results_2 = gwas(geno_df_2, pheno_array_2)
    p_values_2 = gwas_results_2['P-value']
    beta_2 = gwas_results_2['Beta']
    log_p_values_2 = -np.log10(p_values_2)

    # Calculate the correlation between beta value
    df = pd.DataFrame({"Beta1": beta_1, "Beta2": beta_2})
    corr_beta = df["Beta1"].corr(df["Beta2"])
    print("Pearson correlation between Beta:", corr_beta)

    # Set colors for chromosomes
    chrom_colors_1 = ['#17becf', '#9edae5']
    chrom_colors_2 = ['#2AAA8A', '#C1E1C1']
    x_ticks = []
    x_labels = []
    current_position = 0

    # Initialize the plot
    plt.figure(figsize=(14, 8))
    
    # Loop through chromosomes and plot results
    for chrom, last_index in chrom_dict.items():
        start_index = current_position
        end_index = last_index
        chrom_range = np.arange(start_index, end_index)

        # Plot GWAS1 (above the chromosome line)
        plt.scatter(
            chrom_range,
            log_p_values_1[start_index:end_index],
            color=chrom_colors_1[chrom % 2],
            s=10,
            label=f"Chromosome {chrom}" if chrom == 1 else None
        )

        # Plot GWAS2 (below the chromosome line, mirrored)
        plt.scatter(
            chrom_range,
            -log_p_values_2[start_index:end_index],
            color=chrom_colors_2[chrom % 2],
            s=10
        )
        
        # Prepare chromosome labels for the middle of each chromosome
        x_ticks.append((start_index + end_index) // 2)
        x_labels.append(f"Chr{chrom}")

        # Move to the next chromosome
        current_position = end_index

    # Draw a horizontal line for the chromosome axis
    plt.axhline(y=0, color='black', linewidth=2)
    # Add significance threshold line
    threshold = -np.log10(0.05 / len(p_values_1))  # Bonferroni correction
    plt.axhline(y=threshold, color='#e377c2', linestyle='--', linewidth=0.8)
    plt.axhline(y=-threshold, color='#e377c2', linestyle='--', linewidth=0.8)

    # Add population labels
    plt.text(0, max(log_p_values_1) * 0.98, label1, fontsize=16, weight='bold', color="#0047AB")
    plt.text(0, -max(log_p_values_2) * 0.98, label2, fontsize=16, weight='bold', color="#355E3B")
    
    # Beautify the plot
    plt.xticks(x_ticks, x_labels, rotation=90)
    plt.xlabel("Chromosomes", fontsize=12)
    plt.ylabel("-log10(P-value)", fontsize=12)

     # Adjust y-axis labels to be positive both above and below the line
    plt.gca().set_yticks(
        list(plt.gca().get_yticks()) + [-tick for tick in plt.gca().get_yticks() if tick > 0]
    )
    plt.gca().set_yticklabels(
        [f"{abs(tick):.0f}" for tick in plt.gca().get_yticks()]
    )
    # plt.title("Comparison of Two GWAS Results", fontsize=14)
    plt.tight_layout()
    # plt.savefig(save_path_img, format="png")
    plt.savefig(save_path_img, format='eps', dpi=600, bbox_inches='tight')
    plt.show()

    return corr_beta