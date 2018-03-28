import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import statsmodels.api as sm
import itertools
import time
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/Hitters.csv',usecols = range(1,21))

# Drop all rows with Na
data = data.dropna(axis=0)

y = data.Salary

# Create dummy varialbes
dummies = pd.get_dummies(data[['League', 'Division', 'NewLeague']])

# Drop columns with the independent variable and dummy variables
X_ = data.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define feature set X
X = pd.concat([X_,dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

def processSubset(featureSet):
    # Estimate model with the given subset 
    model = sm.OLS(y,X[list(featureSet)])
    regr = model.fit()

    # Calculate RSS of the model
    RSS = ((regr.predict(X[list(featureSet)])-y)**2).sum()
    return {"model":regr, "RSS":RSS}

def getBest(k):
    tic = time.time()
    results = []

    # Create combinations with k predictors and fit all p models (with proccesSubset)
    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))
    
    models = pd.DataFrame(results)

    # Find the best model - the one with the smallest RSS
    best_model = models.loc[models['RSS'].idxmin()]
    toc = time.time()
    print('Number of predictors:', k, '- Elasped time:', round(toc-tic, 1), 'seconds')
    # Return the best model
    return best_model

models = pd.DataFrame(columns=["RSS", "model"])

# Do best subset selection for k=1,2, ... ,p:
for i in range(1, 15):
    models.loc[i]=getBest(i)

#print(models.loc[2, "model"].summary())

plt.figure(figsize=(20, 10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

plt.subplot(2, 2, 1)
plt.plot(models["RSS"])
plt.plot(models["RSS"].idxmin(), models["RSS"].min(), "xr")
plt.xlabel('Number of predictors')
plt.ylabel('RSS')

rsquared = models.apply(lambda row: row[1].rsquared, axis=1)
plt.subplot(2, 2, 2)
plt.plot(rsquared)
plt.plot(rsquared.idxmax(), rsquared.max(), "xr")
plt.xlabel('Number of predictors')
plt.ylabel('Adjusted $R^2$')

aic = models.apply(lambda row: row[1].aic, axis=1)
plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.idxmin(), aic.min(), "xr")
plt.xlabel('Number of predictors')
plt.ylabel('AIC')

bic = models.apply(lambda row: row[1].bic, axis=1)
plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.idxmin(), bic.min(), "xr")
plt.xlabel('Number of predictors')
plt.ylabel('BIC')

#plt.savefig('validationSet.png', dpi=200)