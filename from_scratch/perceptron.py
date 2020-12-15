import numpy as np

def transform_data(features : np.ndarray) -> np.ndarray:
    """
    Transform data.

    Parameters
    ----------
    features (np.ndarray) - input features of shape (# features, # examples)

    Output
    ------
    transformed_features (np.ndarray) - transformed features of shape (# features, # examples)
    """
    return features

class Perceptron():
    """
    A linear perceptron for classification.

    Member Variables
    ----------------
    max_iterations (int) - the number of iterations to run the learning algorithms
    weights (np.ndarray) - array of weights in the linear function of shape (1, # features + 1)
    training_features (np.ndarray) - the features used in training of shape (# features + 1, # examples)
    training_targets (np.ndarray) - the targets used in training of shape (1, # examples)
    """

    def __init__(self, max_iterations : int = 200):
        self.max_iterations = max_iterations
        self.weights = None
        self.training_features = None
        self.training_targets = None

    def fit(self, features : np.ndarray, targets : np.ndarray) -> None:
        """
        Fit a single layer perceptron to classify data as one way or another.

        Parameters
        ----------
        features (np.ndarray) - training features of shape (# features, # examples)
        targets (np.ndarray) - training targets of shape (1, # examples)
        """
        # Assert input shape sizes
        assert(len(features.shape) == 2)
        assert(len(targets.shape) == 2)
        assert(targets.shape[0] == 1)

        # Initialize parameters
        self.weights = np.zeros(features.shape[1] + 1)
        training_features = np.append(np.ones((1, features.shape[1])), features, axis = 0)

        # Train until convergence or max number of iterations is reached
        iterations = 0
        while iterations < self.max_iterations:
            ## Calculating predictions
            g = np.matmul(self.weights, training_features)
            y_hat = (g > 1).astype("int")
            y_hat[y_hat <= 1] = -1

            ## Return if all predictions are correct; otherwise, update self.weights
            accurate_predictions = y_hat == targets
            if np.all(accurate_predictions):
                break
            else:
                self.weights += np.matmul(accurate_predictions, training_features.T)

            iterations += 1
        
        # Save training data
        self.training_features = training_features
        self.training_features = targets

    def predict(self, features : np.ndarray) -> np.ndarray:
        """
        Used the trained model to predict target classes.  Only call this function after calling fit.

        Parameters
        ----------
        features (np.ndarray) - features of shape (# features, # examples)
        
        Output
        ------
        predictions (np.ndarray) - predicted target classes of shape (1, # examples)
        """
        # Prepare features
        testing_features = np.append(np.ones((1, features.shape[1])), features, axis = 0)

        # Calculating predictions
        g = np.matmul(self.weights, testing_features)
        predictions = (g > 1).astype("int")
        predictions[predictions <= 1] = -1

        return predictions