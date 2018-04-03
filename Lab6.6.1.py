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

ridge = Ridge(normalize = True)
coefs = []

# Performe ridge regression for all lambdas in the grid
for a in grid:
    ridge.set_params(alpha = a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
np.shape(coefs)

ax = plt.gca()
ax.plot(grid, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

ridge2 = Ridge(alpha = 4, normalize = True)
ridge2.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred2 = ridge2.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge2.coef_, index = X.columns)) # Print coefficients
print("MSE alpha 4: ", mean_squared_error(y_test, pred2))          # Calculate the test MSE

ridge3 = Ridge(alpha = 10**10, normalize = True)
ridge3.fit(X_train, y_train)             # Fit a ridge regression on the training data
pred3 = ridge3.predict(X_test)           # Use this model to predict the test data
print(pd.Series(ridge3.coef_, index = X.columns)) # Print coefficients
print("MSE alpha 10^10: ", mmean_squared_error(y_test, pred3))          # Calculate the test MSE