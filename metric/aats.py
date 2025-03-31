""" 
Nearest Neighbor Adversarial Accuracy is a metric used to evaluate privacy leakage in generated samples. The idea is to ensure that the generated samples resemble the real ones without being geometrically too close, thereby preserving privacy.

Reference:
-----
Andrew Yale, Saloni Dash, Ritik Dutta, Isabelle Guyon, Adrien Pavao, Kristin P. Bennett,
Generation and evaluation of privacy preserving synthetic health data, Neurocomputing, Volume 416, 2020, Pages 244-255, ISSN 0925-2312,
https://doi.org/10.1016/j.neucom.2019.12.136. (https://www.sciencedirect.com/science/article/pii/S0925231220305117)
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# SEED
np.random.seed(42)

def AATS(real_data, synthetic_data, metric = "manhattan"):
    """
    Compute AATS scores directly from real and synthetic data.
    
    Args:
        real_data (pd.DataFrame): real genetic data (individuals as rows, SNPs as columns).
        synthetic_data (pd.DataFrame): synthetic genetic data (individuals as rows, SNPs as columns).
        metric (string): distance metric used during 1NN model, chose between 'manhattan' and 'euclidean'
    
    Returns:
        A dictionary with:
        - 'AA_real': Adversarial accuracy for the real data.
        - 'AA_synthetic': Adversarial accuracy for the synthetic data.
        - 'AATS': Average of AA_real and AA_synthetic.
    """
    
    def compute_aa(cat_self, cat_other):
        """
        Compute adversarial accuracy for a single category.
        Args:
            cat_self: DataFrame for the current category.
            cat_other: DataFrame for the opposite category.
        Returns:
            Adversarial accuracy score for the category.
        """
        # Self-distances (1-NN, excluding itself)
        nn_self = NearestNeighbors(n_neighbors=1, metric=metric).fit(cat_self.values)
        self_distances = nn_self.kneighbors()[0].flatten() # Minimum distance to another point
        
        # Cross-distances (to the opposite category)
        nn_other = NearestNeighbors(n_neighbors=1, metric=metric).fit(cat_other.values)
        cross_distances = nn_other.kneighbors(cat_self.values)[0].flatten()
        
        # Compute adversarial accuracy
        return np.mean(self_distances < cross_distances)

    # Parallelize the computation of AA for real and synthetic data
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                compute_aa,
                [real_data, synthetic_data],
                [synthetic_data, real_data]
            )
        )

    # Unpack results
    AA_real, AA_synthetic = results

    # Compute AATS score
    AATS_score = {
        'AA_real': AA_real,
        'AA_synthetic': AA_synthetic,
        'AATS': (AA_real + AA_synthetic) / 2
    }

    return AATS_score



def plot_aats(aats_scores, label1, label2, save_path):
    """
    Plot a histogram of AA_real, AA_synthetic, and AATS values.

    Args:
        aats_scores (dict): Dictionary containing 'AA_real', 'AA_synthetic', and 'AATS'.
        label1 (string): label of the first term in atts_scores to be displayed in the plot
        label2 (string): label of the second term in atts_scores to be displayed in the plot
        save_path (string): the path where the pca plot will be saved
    """
    # Extract values and labels
    values = [aats_scores['AA_real'], aats_scores['AA_synthetic'], aats_scores['AATS']]
    labels = [label1, label2, 'AATS']
    
    # Define bar positions
    x_positions = range(len(labels))
    
    # Create the bar plot
    plt.bar(x_positions, values, color=['#b89ac6', '#b8d0e5', '#f7dd83'], alpha=0.9, edgecolor='black')

    # Add value annotations
    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f'{value:.2f}', ha='center', fontsize=10)

    # Add horizontal line at y=0.5
    plt.axhline(y=0.5, color='#d1e7de', linestyle='--', linewidth=2)

    # Set axis labels and title
    plt.xticks(x_positions, labels)
    plt.ylabel('Value')
    plt.ylim(0, 1.1)  # Extend y-axis to accommodate annotations
    #plt.title('Adversarial Accuracy', fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.legend() 
    # plt.savefig(save_path, format="png")
    plt.savefig(save_path+".eps", format='eps', dpi=600, bbox_inches='tight')
    plt.savefig(save_path+".pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()