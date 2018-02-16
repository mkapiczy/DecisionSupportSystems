import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

data = np.loadtxt("Boston.data")
xData = []
yData = []
for item in data:
    xData.append([item[12]]) # The LSTAT Column
    yData.append(item[13]) # The MEDV Column

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(xData, yData)

# Printing relevant values
print('Coefficients:' , regr.coef_)
print('Residuals:' , regr._residues)

print("Mean squared error: %.2f" % np.mean((regr.predict(xData) - yData) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(xData, yData))

# Plot outputs
plt.scatter(xData, yData,  color='black')
plt.plot(xData, regr.predict(xData), color='blue', linewidth=3)
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()