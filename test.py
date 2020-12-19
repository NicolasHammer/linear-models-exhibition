from from_scratch.linear_regression import LinearRegression
from from_scratch.polynomial_regression import PolynomialRegression
from from_scratch.import_data import train_test_split
from from_scratch.perceptron import Perceptron
from from_scratch.metrics import mean_squared_error, accuracy
from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as pltr

features = np.linspace(0, 10, 100)
features = features.reshape((1, features.shape[0]))

weights = np.array([3, -3, 2]).reshape(1, 3)
targets = np.zeros(shape=(1, features.shape[1]))
for degree in range(0, 3):
    targets += weights[0, degree]*(features[0]**degree)

model = PolynomialRegression(degree = 2)
model.fit(features, targets)
model.visualize(features, targets, ["Feature 1", "Feature 2"])