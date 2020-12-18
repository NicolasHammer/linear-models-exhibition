import numpy as np


def mean_squared_error(estimates: np.ndarray, targets: np.ndarray) -> float:
    """
    Return the mean squared error between the estimates and the targets.
    """
    return np.sum(np.power(estimates - targets, 2)/estimates.shape[1])

def confusion_matrix(predictions : np.ndarray, actual : np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    predictions (np.ndarray) - predicted labels\n
    actual (np.ndarray) - actual labels

    Output
    ------
    confusion_matrix (np.ndarray) - 2x2 confusion matric between actual and predictions
    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]
    """
    if predictions.shape != actual.shape:
        raise ValueError("predictions and actual must be the same shape!")

    return np.array([
        [np.sum(np.logical_and(actual == -1, predictions == -1)), np.sum(np.logical_and(actual == -1, predictions == 1))],
        [np.sum(np.logical_and(actual == 1, predictions == -1)), np.sum(np.logical_and(actual == 1, predictions == 1))]
    ])

def accuracy(predictions : np.ndarray, actual : np.ndarray) -> float:
    """
    Parameters
    ----------
    predictions (np.ndarray) - predicted labels\n
    actual (np.ndarray) - actual labels

    Output
    ------
    accuracy (float) - accuracy score
    """
    if predictions.shape != actual.shape:
        raise ValueError("predictions and actual must be the same length!")
    
    conf_mat = confusion_matrix(predictions, actual)

    return float((conf_mat[0, 0] + conf_mat[1, 1])/np.sum(conf_mat))