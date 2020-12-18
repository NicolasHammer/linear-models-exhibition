import numpy as np
import matplotlib.pyplot as plt


def transform_data(features: np.ndarray) -> np.ndarray:
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
    max_iterations (int) - the number of iterations to run the learning algorithms\n
    weights (np.ndarray) - array of weights in the linear function of shape (1, # features + 1)\n
    training_features (np.ndarray) - the features used in training of shape (# features + 1, # examples)\n
    training_targets (np.ndarray) - the targets used in training of shape (1, # examples)
    """

    def __init__(self, max_iterations: int = 200):
        self.max_iterations = max_iterations
        self.weights = None
        self.training_features = None
        self.training_targets = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit a single layer perceptron to classify data as one way or another.

        Parameters
        ----------
        features (np.ndarray) - training features of shape (# features, # examples)\n
        targets (np.ndarray) - training targets of shape (1, # examples)
        """
        # Assert input shape sizes
        assert(len(features.shape) == 2)
        assert(len(targets.shape) == 2)
        assert(targets.shape[0] == 1)

        # Initialize parameters
        self.weights = np.zeros((1, features.shape[0] + 1))
        training_features = np.vstack(
            (np.ones((1, features.shape[1])), features))

        # Train until convergence or max number of iterations is reached
        iterations = 0
        while iterations < self.max_iterations:
            # Calculating predictions
            g = np.matmul(self.weights, training_features)
            y_hat = (g > 1).astype("int")
            y_hat[y_hat == 0] = -1

            # Return if all predictions are correct; otherwise, update self.weights
            incorrect_predictions = targets[0] != y_hat[0]
            if np.any(incorrect_predictions):
                self.weights += np.matmul(targets[0, incorrect_predictions],
                                          training_features[:, incorrect_predictions].T)
            else:
                break
                
            iterations += 1

        # Save training data
        self.training_features = training_features
        self.training_features = targets

    def predict(self, features: np.ndarray) -> np.ndarray:
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
        testing_features = np.vstack(
            (np.ones((1, features.shape[1])), features))

        # Calculating predictions
        g = np.matmul(self.weights, testing_features)
        predictions = (g > 1).astype("int")
        predictions[predictions == 0] = -1

        return predictions

    def visualize(self, features: np.ndarray, targets: np.ndarray, axes_labels: list) -> None:
        """
        Create a visualization for 2D or 3D data that plots a scatterplot for features with colors
        corresponding to the target and the perceptron line/plane.

        Parameters
        ----------
        features (np.ndarray) - features of shape (# features, # examples)\n
        target (np.ndarray) - targets of shape (1, # examples)\n
        axes_labels (list) - labels for axes
        """
        if features.shape[0] == 2:  # 2D Case
            # Syntax for 2D Projection
            ax = plt.axes()

            # Produce scatterplot
            ax.scatter(features[0, targets[0] == 1].T, features[1, targets[0] == 1].T, c='b', marker='o',
                       label="Positive Classification")
            ax.scatter(features[0, targets[0] == -1].T, features[1, targets[0] == -1].T, c='r', marker='o',
                       label="Negative Classification")

            # Produce perceptron line
            w_0 = self.weights[0, 0]
            w_1 = self.weights[0, 1]
            w_2 = self.weights[0, 2]

            perceptron = -(w_1/w_2)*features[0, :] - (w_0/w_2) if w_2 != 0 else np.zeros((features.shape[1],))

            # Plot line
            ax.plot(features[0, :].T, perceptron.T, c='k')

            ax.legend(loc="upper left")
            ax.set_title(
                f"{axes_labels[1]} over {axes_labels[0]} with Decision Boundary")
            ax.set_xlabel(axes_labels[0], fontsize=12)
            ax.set_ylabel(axes_labels[1], fontsize=12)

            # Show figure
            plt.show()
        elif features.shape[0] == 3:  # 3D Case
            pass
        else:
            raise ValueError("Data must be 2D or 3D to visualize.")
