import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    """
    Linear regression from scratch.

    Member Variables
    ----------------
    weights (np.ndarray) - array of weights in the linear funciton of shape 
                           (1, # weights) = (1, # features)\n
    training_features (np.ndarray) - feature data used to fit model\n
    training_targets (np.ndarray) - target data used to fit model
    """

    def __init__(self):
        self.weights = None
        self.training_features = None
        self.training_targets = None

    def fit(self, features : np.ndarray, targets : np.ndarray) -> None:
        """
        Fit our multivariate, linear model to the given data.

        Parameters
        ----------
        features (np.ndarray) - a 2D array of shape (# features, # examples)\n
        targets (np.ndarray) - a 1D array of shape (1, # targets) = (1, # examples)
        """
        # Assert input shape sizes
        assert(len(features.shape) == 2)
        assert(len(targets.shape) == 2)
        assert(targets.shape[0] == 1)

        # Create training features
        training_features = np.vstack((np.ones((1, features.shape[1])), features))

        # Closed form solution of weights: w = (X^{T}X)^{-1}X^{T}y
        self.weights = np.matmul(
            np.matmul(
                np.linalg.pinv(np.matmul(training_features, training_features.T)),
                training_features
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
        features_with_bias = np.vstack((np.ones((1, features.shape[1])), features))
        return np.matmul(self.weights, features_with_bias)

    def visualize(self, features : np.ndarray, targets : np.ndarray, axes_labels : list) -> None:
        """
        Creates visualization for 2D or 3D data that displays a scatterplot and LSRL over the scatterplot.

        Parameters
        ----------
        features (np.ndarray) - 2D or 3D data to be converted to scatterplot
        targets (np.ndarray) - targets to be converted to scatterplot
        axes_labels (list) - labels for axes
        """
        if features.shape[0] == 1: # 2D case
            # Syntax for 2D Projection
            ax = plt.axes()

            # Define line
            minimum = features.min()
            maximum = features.max()
            x = np.linspace(int(minimum), int(maximum), 10)
            x = x.reshape((1, x.shape[0]))

            y = self.predict(x)

            # Plot line and scatterplot
            ax.scatter(features.T, targets.T, c = 'g', marker = 'o')
            ax.plot(x.T, y.T, c = 'r')
            ax.set_title(f"{axes_labels[1]} over {axes_labels[0]} with LSRL")
            ax.set_xlabel(axes_labels[0], fontsize = 12)
            ax.set_ylabel(axes_labels[1], fontsize = 12)
            plt.show()
        elif features.shape [0] == 2: # 3D case
            # Syntax for 3D Projection
            ax = plt.axes(projection = "3d")

            # Define line
            minX = features[0,:].min()
            maxX = features[0,:].max()
            x = np.linspace(int(minX), int(maxX), 10)
            x = x.reshape((1, x.shape[0]))

            minY = features[1,:].min()
            maxY = features[1,:].max()
            y = np.linspace(int(minY), int(maxY), 10)
            y = y.reshape((1, y.shape[0]))

            z = self.predict(np.vstack((x, y)))

            # Plot line and scatterplot
            ax.scatter(features[0,:].T, features[1,:].T, targets.T, c = 'g', marker = 'o')
            ax.plot3D(x.flatten(), y.flatten(), z.flatten(), c = 'r')
            ax.set_title(f"{axes_labels[2]} over ({axes_labels[0]}, {axes_labels[1]}) with LSRL")
            ax.set_xlabel(axes_labels[0], fontsize = 12)
            ax.set_ylabel(axes_labels[1], fontsize = 12)
            ax.set_zlabel(axes_labels[2], fontsize = 12)
            plt.show()
        else:
            raise ValueError("Data must be 2D or 3D to visualize.")