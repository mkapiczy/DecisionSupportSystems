import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn as sns

data = np.loadtxt("Boston.data")

xData = []
yData = []
for item in data:
    xData.append([item[12]]) # The LSTAT Column
    yData.append(item[13]) # The MEDV Column

# Create linear regression object
regr = linear_model.LinearRegression()
model_fit = regr.fit(xData, yData)

# Printing relevant values
print('(Intercept):' , regr.intercept_)
print('Coefficients:' , regr.coef_)
print('Residuals:' , regr._residues)
print('R2 score', regr.score(xData,yData))

print("Mean squared error: %.2f" % np.mean((regr.predict(xData) - yData) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(xData, yData))

# Plot outputs
dataPlot = plt.figure(1)
plt.scatter(xData, yData,  color='black')
plt.plot(xData, regr.predict(xData), color='blue', linewidth=3)
plt.xlabel("LSTAT")
plt.ylabel("MEDV")

# Residuals vs Fitted
resVsFitted = plt.figure(2)
resVsFitted.axes[0] = sns.residplot(regr.predict(xData), regr.predict(xData) + yData, 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
resVsFitted.axes[0].set_title('Residuals vs Fitted')
resVsFitted.axes[0].set_xlabel('Fitted values')
resVsFitted.axes[0].set_ylabel('Residuals')


#Risidual plot
plt.figure(3)
plt.scatter(regr.predict(xData), regr.predict(xData) + yData)
#plt.scatter(regr.predict(xData), regr.predict(xData) - yData, c='b', s=40, alpha=0.5)
plt.show()