from sklearn import datasets
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])

import statsmodels.api as sm

X = df[["LSTAT", "AGE"]]
y = target["MEDV"]

X = sm.add_constant(X)

residual_model = sm.OLS(y, X).fit()
predictions = residual_model.predict(X)

print(residual_model.summary().tables[1])
print("R^2: %f" % residual_model.rsquared)

print("RSE: %f" % np.sqrt(residual_model.mse_resid))

def variance_inflation_factors(exog_df):
    '''
    Parameters
    ----------
    exog_df : dataframe, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression.

    Returns
    -------
    vif : Series
        variance inflation factors
    '''
    exog_df = add_constant(exog_df)
    vifs = pd.Series(
        [1 / (1. - OLS(exog_df[col].values,
                       exog_df.loc[:, exog_df.columns != col].values).fit().rsquared)
         for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    return vifs

print(variance_inflation_factors(df))