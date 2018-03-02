import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('datasets/Smarket.csv', usecols = range(1,10))

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

# Convert probabilities to 'bools'
# 1 is up like before and 0 is down
predictions_nominal = [1 if x > 0.5 else 0 for x in predictions]

from sklearn.metrics import confusion_matrix
true_values = data.Direction
conf_mat = confusion_matrix(true_values, predictions_nominal)
print(conf_mat)

# Probability for correct result
res = (conf_mat[1,1]+conf_mat[0,0])/sum(sum(conf_mat))
print(res)

####  Test and training data ####
# Select training data only
# because it is sorted.
train_data = data[0:sum(data.Year<2005)]
test_data = data[sum(data.Year<2005):]

# Create the train model
res_train = smf.glm("Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume",data=train_data,family=sm.families.Binomial()).fit()
# Summary of the trained model
print(res_train.summary().tables[1])
# Get the train models coefficients
print(res_train.params)

predictions = res_train.predict(test_data)
predictions_nominal = [1 if x > 0.5 else 0 for x in predictions]

true_values = test_data.Direction
conf_mat_test = confusion_matrix(predictions_nominal, true_values)
print(conf_mat_test)

# Probability for correct result
res_test = (conf_mat_test[1,1]+conf_mat_test[0,0])/sum(sum(conf_mat_test))
print(res_test)

####  NOW ONLY LAG 1 and 2 ####
# Select training data only
# because it is sorted.
train_data = data[0:sum(data.Year<2005)]
test_data = data[sum(data.Year<2005):]

# Create the train model
res_train = smf.glm("Direction ~ Lag1 + Lag2",data=train_data,family=sm.families.Binomial()).fit()
# Summary of the trained model
print(res_train.summary().tables[1])
# Get the train models coefficients
print(res_train.params)

predictions = res_train.predict(test_data)
predictions_nominal = [1 if x > 0.5 else 0 for x in predictions]

true_values = test_data.Direction
conf_mat_test = confusion_matrix(predictions_nominal, true_values)
print(conf_mat_test)

# Probability for correct result
res_test = (conf_mat_test[1,1]+conf_mat_test[0,0])/sum(sum(conf_mat_test))
print(res_test)


print(res_train.predict(pd.DataFrame([[1.2, 1.1], [1.5, -0.8]], columns = ["Lag1", "Lag2"])))