import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import pandas as pd
import statsmodels.formula.api as smf

data = pd.read_csv('datasets/Auto.csv')
# Extract columns and reshape np array to use with LinearRegression Fit
x = data['mpg'].values.reshape(-1, 1)
y = data['horsepower'].values.reshape(-1, 1)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.50,random_state=1)

# Fit, predict and calculate mean squared error
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)

print ('MSE for mpg and horsepower is: ', MSE)
