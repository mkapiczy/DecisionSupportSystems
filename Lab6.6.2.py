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

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []

for a in grid:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
ax = plt.gca()
ax.plot(grid*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
# plt.show()

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X_train, y_train)

lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)
print ("MSE: ", mean_squared_error(y_test, lasso.predict(X_test)))

print (pd.Series(lasso.coef_, index=X.columns))