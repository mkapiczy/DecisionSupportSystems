import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
from sklearn.preprocessing import normalize
import math

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
resVsFitted.axes[0] = sns.residplot(regr.predict(xData), yData - regr.predict(xData), 
                          lowess=True, 
                          scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
resVsFitted.axes[0].set_title('Residuals vs Fitted')
resVsFitted.axes[0].set_xlabel('Fitted values')
resVsFitted.axes[0].set_ylabel('Residuals')

#QQ Plot
residuals = (yData - regr.predict(xData))
standardized_residuals = residuals/np.std(residuals)
QQ = ProbPlot(standardized_residuals)
qqPlot = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
qqPlot.axes[0].set_title('Normal Q-Q')
qqPlot.axes[0].set_xlabel('Theoretical Quantiles')
qqPlot.axes[0].set_ylabel('Standardized Residuals');
# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(standardized_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]
for r, i in enumerate(abs_norm_resid_top_3):
    qqPlot.axes[0].annotate(i, 
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   standardized_residuals[i]));

#Risidual plot
plt.figure(4)
plt.scatter(regr.predict(xData), regr.predict(xData) + yData)
#plt.scatter(regr.predict(xData), regr.predict(xData) - yData, c='b', s=40, alpha=0.5)
plt.show()