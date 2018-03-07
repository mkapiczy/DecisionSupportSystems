import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data = pd.read_csv('datasets/Smarket.csv', usecols = range(1,10), parse_dates=True)

x_train = data[0:sum(data.Year<2005)][['Lag1', 'Lag2']]
y_train = data[0:sum(data.Year<2005)]['Direction']

lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(x_train, y_train)

print("Down : %f" % lda.priors_[0])
print("Up : %f" % lda.priors_[1])

# Group mean and coeffs
means = pd.DataFrame(lda.means_,['Down', 'Up'], ['Lag1', 'Lag2'])
coeffs = pd.DataFrame(lda.scalings_, ['Lag1', 'Lag2'], ['LD'])
print(means)
print(coeffs)
y_predicted = lda.predict(x_train)

pd.DataFrame(confusion_matrix(y_train, y_predicted).T,['Down', 'Up'], ['Down', 'Up'])
# Create function to convert a report to dictionary
def reportToDictionary(classificationReport):
    tmp = list()
    for row in classificationReport.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    # Store in dictionary
    measures = tmp[0]

    classData = defaultdict(dict)
    for row in tmp[1:]:
        label = row[0]
        for i, j in enumerate(measures):
            classData[label][j.strip()] = float(row[i + 1].strip())
    return classData

classificationReport = classification_report(y_train, y_predicted, digits=3)
dictionary = pd.DataFrame(reportToDictionary(classificationReport)).T
print(dictionary)

x_test = data[sum(data.Year<2005):][['Lag1', 'Lag2']]
y_test = data[sum(data.Year<2005):][['Direction']]
y_pred = lda.predict(x_test)

pd.DataFrame(confusion_matrix(y_test, y_pred).T, ['Down', 'Up'], ['Down', 'Up'])
classificationReport = classification_report(y_test, y_pred, digits=3)
newDictionary = pd.DataFrame(reportToDictionary(classificationReport)).T
print(newDictionary)

predictedP = lda.predict_proba(x_test)
print(sum(predictedP[: ,0] >= 0.5))
print(sum(predictedP[: ,0] < 0.5))
# The latter is the largest which means the model corresponds to the probability of the market will decrease.

pd.DataFrame(predictedP[10:20, 0], y_pred[10:20]).T
print(sum(predictedP[:, 0] > 0.9))
print(max(predictedP[:,0]))
# This means that no days in 2005 meets the threshold. The greates posterior probability in 2005 was 52.02 percent.
# Could be done for more combinations: eg. 1, 2 and volume || 1, 2 and 3 ||
