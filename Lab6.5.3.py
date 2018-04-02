import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv("datasets/Hitters.csv",usecols = range(1,21))

####### REUSE FROM PREVIUS LAB
# Drop all rows with Na
data = data.dropna(axis=0)
# Extract independent variable Salary
y = pd.DataFrame(data.Salary)


# Create dummy variables
dummies = pd.get_dummies(data[["League", "Division", "NewLeague"]])
# Drop already used data for dummy and the independent variable
X_ = data.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define feature set X
X = pd.concat([X_,dummies[["League_N", "Division_W", "NewLeague_N"]]], axis=1)

# To use the validation set approach we begin by split observations into test and train.
# Done by creating a random vector of elements equal to true if the observation in train is false.

np.random.seed(seed=12)
train = np.random.choice([True, False], size = len(y), replace = True)
# The test will just be the inverted set of train
test = np.invert(train)

# We modify the helper function from previous as we need to take both train and test sets
def processSubset(feature_set, Xtrain, Ytrain, Xtest, Ytest):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(Ytrain,Xtrain[list(feature_set)])
    RSS = ((model.fit().predict(Xtest[list(feature_set)]) - Ytest) ** 2).sum()
    return {"Model":model.fit(), "RSS":RSS}


# Forward selection
def forward(predictors, Xtrain, Ytrain, Xtest, Ytest):
    #Init results
    results = []
    missing_predictors = [predictor for predictor in Xtrain.columns if predictor not in predictors]
    
    for predictor in missing_predictors:
        results.append(processSubset(predictors+[predictor], Xtrain, Ytrain, Xtest, Ytest))
    
    models = pd.DataFrame(results)
    # Return the best model
    return models.loc[models['RSS'].argmin()]

# Validation
#Creating dataframe holding RSS and model
forward_models = pd.DataFrame(columns=["RSS", "Model"])

predictors = []

for i in range(0,len(X.columns)):    
    forward_models.loc[i+1] = forward(predictors, X[train], y[train]["Salary"], X[test], y[test]["Salary"])
    predictors = forward_models.loc[i+1]["Model"].model.exog_names


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
    forward_models_test.loc[i+1] = forward(predictors, X[test], y[test]["Salary"], X[test], y[test]["Salary"])
    predictors = forward_models_test.loc[i+1]["Model"].model.exog_names

# Full data set has different set of predictors that ten variable. Which can be seen in plot
print("Full model")
print(forward_models.loc[10, "Model"].model.exog_names)
print("Ten predictors")
print(forward_models_test.loc[10, "Model"].model.exog_names)


# Cross validation
# We use k = 10 folds, and creates dataframe to store

k = 10
np.random.seed(seed=1)
folds = np.random.choice(k, size = len(y), replace = True)

#Dummy array for upcoming data
cross_validation_errors = pd.DataFrame(columns=range(1,k+1), index=range(1,20))
cross_validation_errors = cross_validation_errors.fillna(0)

models_cross_validation = pd.DataFrame(columns=["RSS", "Model"])

# Now a for loop that runs over each fold, in which predictors is reset and every element is run through
for j in range(0,k):
    predictors = []
    for i in range(0,len(X.columns)):
        # Then perform forward selection on the full dataset
        models_cross_validation.loc[i+1] = forward(predictors, X[folds != j], y[folds != j]["Salary"], X[folds == j], y[folds == j]["Salary"])
        cross_validation_errors[j+1][i+1] = models_cross_validation.loc[i+1]["RSS"]
        predictors = models_cross_validation.loc[i+1]["Model"].model.exog_names

# now it is filled up in cross_validation_errors meaning that i, j element is test MSE for ith cross validation
cross_validation_mean = cross_validation_errors.apply(np.mean, axis=1)

plt.plot(cross_validation_mean)
plt.xlabel("# Predictors")
plt.ylabel("CV Error")
plt.plot(cross_validation_mean.argmin(), cross_validation_mean.min(), "or")
plt.show()

# The 9th predictor is the best
print("9th Predictor model")
print(models_cross_validation.loc[9, "Model"].model.exog_names)


