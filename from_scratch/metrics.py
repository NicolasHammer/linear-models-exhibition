import numpy as np

def mean_squared_error(estimates : np.ndarray, targets : np.ndarray) -> float:
    """
    Return the mean squared error between the estimates and the targets.
    """
    return np.power(estimates - targets, 2)/estimates.shape[1]