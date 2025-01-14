import pandas as pd

def basic_sanity_check(synthetic_data, training_data):
    """
    Perform basic sanity checks on the synthetic dataset.
    
    Args:
        synthetic_data: pd.DataFrame, synthetic dataset.
        training_data: pd.DataFrame, training dataset.
    
    Returns:
        dict: A dictionary with results of the sanity checks:
            - 'redundant_rows': Number of redundant rows in the synthetic dataset.
            - 'shared_rows': Number of rows shared between synthetic and training datasets.
    """
    # Check for redundant rows in the synthetic dataset
    redundant_rows = synthetic_data.duplicated(keep='first').sum()

    # Check for rows shared between synthetic and training datasets
    shared_rows = pd.merge(synthetic_data, training_data, how='inner').shape[0]

    # Return results
    return {
        'redundant_rows': redundant_rows,
        'shared_rows': shared_rows
    }