import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import statsmodels.api as sm
import itertools
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

### BACKWARD SELECTION FUNCTION
def backward(predictors):
    result = []
    
    #Try every possible combination
    for p in itertools.combinations(predictors, len(predictors)-1):
        subset = processSubset(p)
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


### BACKWARD TEST
backward_models = pd.DataFrame(columns=["RSS", "Model"])
predictors = X.columns
i = len(predictors)-1
while(i > 0):
    backward_models.loc[i] = backward(predictors)
    predictors = backward_models.loc[i]["Model"].model.exog_names
    i = i - 1

print("BACKWARD, 7 predictors:")
print(backward_models.loc[7, "Model"].summary())