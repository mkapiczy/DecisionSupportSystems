import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('Smarket.csv', usecols = range(1,10))

for x in range(0, data.Direction.size):
  if data.Direction[x].lower() in ['up']: 
    data.loc[x, 'Direction'] = 1
  else:
    data.loc[x, 'Direction'] = 0

res = smf.glm("Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume",data,family=sm.families.Binomial()).fit()
print(res.summary().tables[1])

# print(data.describe().transpose())