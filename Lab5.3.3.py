import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('datasets/Auto.csv')

x = data['horsepower'].values.reshape(-1, 1)
y = data['mpg'].values.reshape(-1, 1)

regr = LinearRegression()
regr.fit(x, y)

print("Coeffecients: %.2f" % regr.coef_)

#Running cross_val_score for polynomial with different degrees
for i in range(0,10):
    poly = PolynomialFeatures(degree=i+1)
    X_train_poly = poly.fit_transform(x)
    lm = linear_model.LinearRegression()
    # Do K-fold cross validation
    k_fold = KFold(n_splits=x.shape[0]) 
    score = cross_val_score(lm, X_train_poly, y, cv=k_fold, scoring = 'neg_mean_squared_error')
    print ('MSE for order:', i+1, score.mean())