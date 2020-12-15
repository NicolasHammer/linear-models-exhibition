import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression():
    """
    Polynomial regression from scratch.

    Member Variables
    ----------------
    degree (int) - the number of degrees in the model
    weights (np.ndarray) - array of weights in the linear funciton of shape 
                           (1, # weights) = (1, # features)\n
    features (np.ndarray) - feature data used to fit model of shape (# features, # examples)\n
    targets (np.ndarray) - target data used to fit model of shape (1, # targets)
    """

    def __init__(self, degree = int):
        self.degree = degree
        self.weights = None
        self.training_features = None
        self.training_targets = None

    def fit(self, features : np.ndarray, targets : np.ndarray) -> None:
        """
        Fit our multivariate, polynomial model to the given data.

        Parameters
        ----------
        features (np.ndarray) - a 1D array of shape (1, # examples)\n
        targets (np.ndarray) - a 1D array of shape (1, # targets) = (1, # examples)
        """
        # Assert input shape sizes
        assert(len(features.shape) == 2)
        assert(features.shape[0] == 1)

        assert(len(targets.shape) == 2)
        assert(targets.shape[0] == 1)

        # Create polynomial_features
        polynoial_features = np.ndarray((self.degree, features.shape[1]))
        polynoial_features[0] = features
        for degree in range(2, degree):
            polynomial_features[degree] = features**degree

        # Closed form solution of weights: w = (X^{T}X)^{-1}X^{T}y
        self.weights = np.matmul(
            np.matmul(
                np.lingalg.pinv(np.matmul(polynoial_features, polynoial_features.T)),
                polynoial_features
                ),
            targets.T
            ).T

        self.training_features = features
        self.training_targets = targets

    def predict(self, features : np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        features (np.ndarray) - a 2D array of shape (# features, # examples)

        Output
        ------
        predictions (np.ndarray) - a 1D array of shape (1, # predictions) = (1, # examples)
        """
        predictions = np.ndarray(shape = (1, features.shape[1]))
        for degree in range(0, self.degree):
            predictions += self.weights[degree]*(features**degree)
        return predictions

    def visualize(self, features : np.ndarray, targets : np.ndarray) -> None:
        """
        Creates visualization for 2D data that displays a scatterplot and LSRL over the scatterplot.

        Parameters
        ----------
        features (np.ndarray) - 2D or 3D data to be converted to scatterplot
        targets (np.ndarray) - targets to be converted to scatterplot
        """
        if features.shape[0] == 1: # 2D case
            # Syntax for 2D Projection
            ax = plt.axes(projection = "2d")

            # Define line
            minimum = features.min()
            maximum = features.max()
            x = np.linspace(minimum, maximum, 100*(maximum - minimum))
            y = self.predict(features)

            # Plot line and scatterplot
            ax.scatter(features.T, targets.T, c = 'g', marker = 'o')
            ax.plot(x, y, c = 'r')
            ax.set_title("Target Data over Feature Data Overlayed with Fitted Polynomial")
            ax.set_xlabel("Feature 1", fontsize = 16)
            ax.set_ylabel("Output", fontsize = 16)
            plt.show()
        else:
            raise ValueError("Data must be 2D to visualize.")