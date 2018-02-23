from sklearn import datasets
import pandas as pd
import numpy as np
data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])

import statsmodels.api as sm

X = df[["LSTAT", "AGE", 'DIS']]
y = target["MEDV"]

X = sm.add_constant(X)

residual_model = sm.OLS(y, X).fit()
predictions = residual_model.predict(X)

print(residual_model.summary().tables[1])
print("R^2: %f" % residual_model.rsquared)

print("RSE: %f" % np.sqrt(residual_model.mse_resid))

