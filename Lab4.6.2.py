import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('Smarket.csv', usecols = range(1,10))

# To get the values to fit the answer
for x in range(0, data.Direction.size):
  if data.Direction[x].lower() in ['up']: 
    data.loc[x, 'Direction'] = 1
  else:
    data.loc[x, 'Direction'] = 0

res = smf.glm("Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume",data,family=sm.families.Binomial()).fit()

# Summary of the model
print(res.summary().tables[1])
# Get the coefficients
print(res.params)

# Get the first 10 predictions 
predictions = res.predict()
print(predictions[0:10])




# print(data.describe().transpose())