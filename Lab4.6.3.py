import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


data = pd.read_csv('Smarket.csv', usecols = range(1,10), parse_dates=True)

x_train = data[0:sum(data.Year<2005)][['Lag1', 'Lag2']]
y_train = data[0:sum(data.Year<2005)]['Direction']

lda = LinearDiscriminantAnalysis(solver='svd')
lda.fit(x_train, y_train)

print("Down : %f" % lda.priors_[0])
print("Up : %f" % lda.priors_[1])

# Group mean and coeffs
pd.DataFrame(lda.means_,['Down', 'Up'], ['Lag1', 'Lag2'])
pd.DataFrame(lda.scalings_, ['Lag1', 'Lag2'], ['LD'])

y_predicted = lda.predict(x_train)

pd.DataFrame(confusion_matrix(y_train, y_predicted).T,['Down', 'Up'], ['Down', 'Up'])
