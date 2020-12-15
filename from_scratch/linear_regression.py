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
        training_features = np.append(np.ones((1, features.shape[1])), features, axis = 0)

        # Closed form solution of weights: w = (X^{T}X)^{-1}X^{T}y
        self.weights = np.matmul(
            np.matmul(
                np.lingalg.pinv(np.matmul(training_features, training_features.T)),
                training_features
                ),
            targets.T
            ).T

        self.training_features = training_features
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
        return np.matmul(self.weights, features)

    def visualize(self, features : np.ndarray, targets : np.ndarray) -> None:
        """
        Creates visualization for 2D or 3D data that displays a scatterplot and LSRL over the scatterplot.

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
        elif features.shape [0] == 2: # 3D case
            # Syntax for 3D Projection
            ax = plt.axes(projection = "3d")

            # Define line
            minX = features[0,:].min()
            maxX = features[0,:].max()
            x = np.linspace(minX, maxX, 100*(maxX - minX))

            minY = features[1,:].min()
            maxY = features[1,:].max()
            y = np.linspace(minY, maxY, 100*(maxY - minY))

            z = self.predict(features)

            # Plot line and scatterplot
            ax.scatter(features[0,:].T, features[1,:].T, targets.T, c = 'g', marker = 'o')
            ax.plot3D(x, y, z, c = 'r')
            ax.set_title()
            ax.set_title("Target Data over Feature Data Overlayed with Fitted Polynomial")
            ax.set_xlabel("Feature 1", fontsize = 16)
            ax.set_ylabel("Feature 2", fontsize = 16)
            ax.set_zlabel("Output", fontsize = 16)
            plt.show()
        else:
            raise ValueError("Data must be 2D or 3D to visualize.")

