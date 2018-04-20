import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error

# Read data and drop rows with NA
df = pd.read_csv('datasets/Hitters.csv').dropna()
# Load Salary into array
y = df.Salary

# Drop the column with the independent variable (Salary)
# And the unnamed name column (Called 'Unnamed: 0' when imported with Pandas)
# And other columns with string variables
X = df.drop(['Unnamed: 0', 'Salary', 'Division', 'League', 'NewLeague'], axis = 1)

# Construct grid from 10^10 to 10^-2
grid = 10**np.linspace(10,-2,100)*0.5

# Perform ridge regression for all lambdas in the grid
ridge = Ridge(normalize = True)
coefs = []
for a in grid:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# Print the shape of the coefficients and plot them    
# print(np.shape(coefs))
# ax = plt.gca()
# ax.plot(grid, coefs)
# ax.set_xscale('log')
# plt.axis('tight')
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.show()

# Split data into 50/50 train/test
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# Fit a ridge regression model with lambda 4
ridge4 = Ridge(alpha = 4, normalize = True)
ridge4.fit(X_train, y_train)                                # Fit a ridge regression on the training data
pred = ridge4.predict(X_test)                               # Use this model to predict the test data
# print(pd.Series(ridge4.coef_, index = X.columns))           # Print coefficients
print("MSE alpha 4: ", round(mean_squared_error(y_test, pred),2))    # Calculate the test MSE

# Fit a ridge regression model with lambda 10^10
ridge1010 = Ridge(alpha = 10**10, normalize = True)
ridge1010.fit(X_train, y_train)
pred = ridge1010.predict(X_test)
# print(pd.Series(ridge1010.coef_, index = X.columns))
print("MSE alpha 10^10: ", round(mean_squared_error(y_test, pred),2))

# Fit a ridge regression model with lambda 0 (Which is equivalent to least squares)
ridge = Ridge(alpha = 0, normalize = True)
ridge.fit(X_train, y_train)
pred = ridge.predict(X_test)
# print(pd.Series(ridge.coef_, index = X.columns))
print("MSE alpha 0 (Least squares): ", round(mean_squared_error(y_test, pred),2))

# Cross-validated ridge regression function
ridgecv = RidgeCV(alphas = grid, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(X_train, y_train)
print("Ridgecv alpha: ", ridgecv.alpha_)

# Find MSE of this value of alpha
ridgeA = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridgeA.fit(X_train, y_train)
print("MSE ridgecv.alpha: ", round(mean_squared_error(y_test, ridgeA.predict(X_test)),2))

# Fit ridge regression model on the full data set
ridgeA.fit(X, y)
print(pd.Series(ridgeA.coef_, index = X.columns))