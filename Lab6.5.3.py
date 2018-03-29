import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/Hitters.csv',usecols = range(1,21))

####### REUSE FROM PREVIUS LAB
# Drop all rows with Na
data = data.dropna(axis=0)
y = data.Salary

# Create dummy varialbes
dummies = pd.get_dummies(data[['League', 'Division', 'NewLeague']])

# Drop columns with the independent variable and dummy variables
X_ = data.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define feature set X
X = pd.concat([X_,dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# To use the calidation set approach we begin by split observations into test and train.
# Done by creating a random bector of elements equal to true if the observation in train is false.

np.random.seed(seed=12)
train = np.random.choice([True, False], size = len(y), replace = True)
test = np.invert(train)

# First naive try, copied forward from 6.5.2
def processSubset(featureSet):
    # Estimate model with the given subset 
    model = sm.OLS(y,X[list(featureSet)])
    regr = model.fit()

    # Calculate RSS of the model
    RSS = ((regr.predict(X[list(featureSet)])-y)**2).sum()
    return {"Model":regr, "RSS":RSS}


### FORWARD SELECTION FUNCTION
def forward(predictors):
    #Get not added predictors using list comprehension
    missing_predictors = [predictor for predictor in X.columns if predictor not in predictors]

    result = []
    
    for p in missing_predictors:
        addedPredictor = predictors + [p]
        subset = processSubset(addedPredictor)
        result.append(subset) 
    
    models = pd.DataFrame(result)

    #Get best RSS
    return models.loc[models['RSS'].argmin()]

### FORWARD TEST
#Creating dataframe holding RSS and model
forward_models = pd.DataFrame(columns=["RSS", "Model"])

predictors = []

for i in range(0,len(X.columns)):
    forward_models.loc[i] = forward(predictors)
    predictors = forward_models.loc[i]["Model"].model.exog_names

#Print result for first 7 predictors
#for i in range(0,7):
print("Predictor " , i+1)
print(forward_models.loc[i, "Model"].summary())
print("FORWARD, 7 predictors:")
print(forward_models.loc[6, "Model"].summary())

plt.plot(forward_models["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')
plt.plot(forward_models["RSS"].argmin(), forward_models["RSS"].min(), "or")
plt.show()

# The model shows that the best is the one with 10 predictors
# Then we perform best subset selection og the 10 predictor model. 
# Here it is important to use the full dataset in order to make accurate estimates.
# 

forward_models_test = pd.DataFrame(columns=["RSS", "Model"])

predictors = []

for i in range(0,10):
    forward_models_test.loc[i+1] = forward(predictors)
    predictors = forward_models_test.loc[i+1]["Model"].model.exog_names

# Full data set has different set of predictors thatn ten variable. Which can be seen in plot

print(forward_models.loc[10, "model"].model.exog_names)
print(forward_models_test.loc[10, "model"].model.exog_names)