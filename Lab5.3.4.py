import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

#Helper functions
def calc_alpha(X,Y):
    return ((np.var(Y) - np.cov(X,Y)) / (np.var(X) + np.var(Y) - 2*np.cov(X,Y)))

#bootstrap function using 1000 repetitions
def bootstrap(data):
    N_rep = 1000;
    total_alpha = 0
    alphaMeans = []
    for i in range(0,N_rep):
        sample = data.sample(frac = 1, replace = True)
        X = sample.X[0:100]
        Y = sample.Y[0:100]
        currentAlpha = calc_alpha(X,Y)
        alphaMeans.append(currentAlpha.mean())
        
    mean = np.mean(alphaMeans)
    std_err = stats.sem(alphaMeans)
        
    return AlphaClass(mean, std_err)
    #return total_alpha / N_rep

#Bootstrap function used for linear regression model using 1000 repetetions 
def bootstrap_lin_reg(x,y):
    N_rep = 1000
    intercepts = []
    coefs = []
    for i in range(0,N_rep):
        #Pick 1000 samples from x and y
        xSample, ySample = resample(x, y, n_samples=1000)
        clf = regr.fit(xSample,ySample)
        intercepts.append(clf.intercept_[0]) 
        coefs.append(clf.coef_[0][0])
    
    return LinearRegressionClass(np.mean(intercepts), np.mean(coefs), stats.sem(intercepts), stats.sem(coefs))

class AlphaClass:
    def __init__(self, mean, std_err):
        self.mean = mean
        self.std_err = std_err

class LinearRegressionClass:
    def __init__(self, intercept, coef, intercept_std_error, coef_std_error):
        self.intercept = intercept
        self.coef = coef
        self.intercept_std_error = intercept_std_error
        self.coef_std_error = coef_std_error

### Estimating the Accuracy of STATISTIC OF INTEREST
portfolio_data = pd.read_csv('datasets/Portfolio.csv')

#Estimate alpha using all 100 observations
X = portfolio_data.X[0:100]
Y = portfolio_data.Y[0:100]
print("Portfolio - Estimate alpha using all samples:")
print(calc_alpha(X,Y).mean())

#Estimate alpha using 100 of the samples with replacement (using sample function)
pf_sample = portfolio_data.sample(frac = 1, replace = True)
X = pf_sample.X[0:100]
Y = pf_sample.Y[0:100]
print("Portfolio - 100 samples with replacement:")
print(calc_alpha(X,Y).mean())

#Bootstrap
print("Portfolio - Bootstrap:")
print("Mean: " , bootstrap(portfolio_data).mean)
print("std_err: " , bootstrap(portfolio_data).std_err)

### Estimating the Accuracy of a LINEAR REGRESSION MODEL
data = pd.read_csv('datasets/Auto.csv')

# Extract columns and reshape np array to use with LinearRegression Fit
x = data['horsepower'].values.reshape(-1, 1)
y = data['mpg'].values.reshape(-1, 1)

#Create Linear Regression Model
regr = LinearRegression()
regr.fit(x, y)
print('(Intercept):', regr.intercept_[0])
print('Coefficients:', regr.coef_[0][0])

#Using bootstrap to estimate intercept and slope terms
result =  bootstrap_lin_reg(x,y)
print('(Intercept) - bootsrap:', result.intercept, ' std. err: ', result.intercept_std_error)
print('Coefficients - bootstrap:', result.coef, ' std. err: ', result.coef_std_error)


