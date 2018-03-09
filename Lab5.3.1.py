import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

data = pd.read_csv('datasets/Auto.csv')
# Extract columns and reshape np array to use with LinearRegression Fit
x = data['horsepower'].values.reshape(-1, 1)
y = data['mpg'].values.reshape(-1, 1)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.50,random_state=2)

# Fit, predict and calculate mean squared error for linear regression
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
print ('MSE for linear regression using mpg and horsepower is:', MSE)

# Fit, predict and calculate mean squared error for polynomial regression
for i in range(0,4):
  pipeline = Pipeline([
      ('poly', PolynomialFeatures(degree=i+2)),
      ('linreg', LinearRegression())
      ])

  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)
  MSE = mean_squared_error(y_test, y_pred)
  print ('MSE for polynomial regression with degree', i+2, 'using mpg and horsepower is:', MSE)
