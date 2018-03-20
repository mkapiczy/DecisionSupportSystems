import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv('datasets/Auto.csv')

x = data['horsepower'].values.reshape(-1, 1)
y = data['mpg'].values.reshape(-1, 1)

regr = LinearRegression()
regr.fit(x, y)

print("Coeffecients: %.2f" % regr.coef_)
#K fold cross validation strategy
knn = KFold(n_splits=10)

scores = list()
xplot = list()
# Fit, predict and calculate mean squared error for polynomial regression for order degree 1-10
for i in range(0,10):
      poly = PolynomialFeatures(degree=i+1)
      x_current = poly.fit_transform(x)
      model = regr.fit(x_current, y)
      #Score using leave one out as split strategy
      score = cross_val_score(model, x_current, y, cv=knn, scoring="neg_mean_squared_error")
      print ('MSE for order:', i+1, score.mean())
      scores.append(round(-score.mean(),2))
      xplot.append(i+1)

plt.tight_layout()
plt.plot(xplot, scores, '-o')
plt.xlabel('Polynomial Degree') 
plt.ylabel ('Mean Squared Error')
plt.savefig('KFOLDCV_533.png', dpi=200)