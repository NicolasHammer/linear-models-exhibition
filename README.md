# Linear Models Exhibition
An exhibition of three linear models built from scratch: 
  1. multiple linear regression
  2. polynomial regression
  3. a perceptron

Each of these models are compared against their respective scikit-learn counterpart: 
  1. sklearn.linear_model.LinearRegression
  2. sklearn.preprocessing.PolynomialFeatures + sklearn.linear_model.LinearRegression
  3. sklearn.linear_model.Perceptron

The data used to illustrate the linear models are, respectively:
  1. [Advertising.csv](https://statlearning.com/data.html) from "An Introduction to Statistical Learning with Applications in R" by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.
  2. [bluegills.csv](https://online.stat.psu.edu/stat462/sites/onlinecourses.science.psu.edu.stat462/files/data/bluegills/index.txt) from (Cook and Weisberg, 1999).
  3. Self-generated clusters using sklearn.datasets.make_blobs()
