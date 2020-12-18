from from_scratch.linear_regression import LinearRegression
from from_scratch.polynomial_regression import PolynomialRegression
from from_scratch.import_data import load_data, train_test_split
from from_scratch.perceptron import Perceptron
from from_scratch.metrics import mean_squared_error, accuracy
from sklearn.datasets import make_blobs

# Generate 2D data
centers = [(3, 2), (-3, -5)]
cluster_std = [1.75, 1.75]

features, targets = make_blobs(n_samples = 100, cluster_std = cluster_std, centers = centers, n_features = 2)
features = features.T
targets = targets.reshape(1, targets.shape[0])
targets[0, targets[0] == 0] = -1

train_features, train_targets, test_features, test_targets = train_test_split(features, targets)

model = Perceptron()
model.fit(train_features, train_targets)
model.visualize(train_features, train_targets, ["Feature 1", "Feature 2"])