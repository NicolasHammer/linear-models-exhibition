from from_scratch.linear_regression import LinearRegression
from from_scratch.import_data import load_data, train_test_split

import csv
import numpy as np

reader = csv.reader(open("linear_regression_dataset.csv", 'r'))

# Extract headers
feature_names = list(next(reader)) # should be list of strings

# Extract features/targets
data = np.array(list(reader)).T # array including features and targets
features = data[1:3].reshape(2, data.shape[1]).astype('float')
targets = data[3].reshape(1, data.shape[1]).astype('float')

model = LinearRegression()
model.fit(features, targets)
model.visualize(features, targets, feature_names[1:4])