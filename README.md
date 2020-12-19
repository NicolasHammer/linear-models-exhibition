# Linear Models Exhibition
An exhibition of three linear models built from scratch: 
<ol>
  <li>multiple linear regression</li>
  <li>polynomial regression</li>
  <li>a perceptron</li>
</ol>
Each of these models are compared against their respective scikit-learn counterpart: 
<ol>
  <li>sklearn.linear_model.LinearRegression</li>
  <li>sklearn.preprocessing.PolynomialFeatures + sklearn.linear_model.LinearRegression</li>
  <li>sklearn.linear_model.Perceptron.</li>
</ol>
The data used to illustrate the linear models are, respectively:
<ol>
  <li> [Advertisting.csv](http://faculty.marshall.usc.edu/gareth-james/ISL/Advertising.csv) from "An Introduction to Statistical Learning with Applications in R" by 
  Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani</li>
  <li> [bluegills.csv](https://online.stat.psu.edu/stat462/sites/onlinecourses.science.psu.edu.stat462/files/data/bluegills/index.txt) from (Cook and Weisberg, 1999)</li>
  <li>  Self-generated clusters using sklearn.datasets.make_blobs()
</ol>