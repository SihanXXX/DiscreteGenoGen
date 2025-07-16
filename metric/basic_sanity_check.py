import pandas as pd

def basic_sanity_check(synthetic, real):
    """
    Perform basic sanity checks on the synthetic dataset.
    
    Args:
        synthetic: pd.DataFrame, synthetic dataset.
        real: pd.DataFrame, real dataset.
    
    Returns:
        dict: A dictionary with results of the sanity checks:
            - 'redundant_rows': Number of redundant rows in the synthetic dataset.
            - 'shared_rows': Number of rows shared between synthetic and real datasets.
    """
    # Check for redundant rows in the synthetic dataset
    redundant_rows = synthetic.duplicated(keep='first').sum()

    # Check for rows shared between synthetic and real datasets
    shared_rows = pd.merge(synthetic, real, how='inner').shape[0]

    # Return results
    return {
        'redundant_rows': redundant_rows,
        'shared_rows': shared_rows
    }