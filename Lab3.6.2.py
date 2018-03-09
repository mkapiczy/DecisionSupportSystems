import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
from statsmodels.graphics.gofplots import ProbPlot
from sklearn import datasets
import pandas as pd
import numpy as np

data = datasets.load_boston()

df = pd.DataFrame(data.data, columns=data.feature_names)

target = pd.DataFrame(data.target, columns=["MEDV"])

import statsmodels.api as sm

xData = df[["LSTAT"]]
yData = target["MEDV"]

X = xData
y = yData

X = sm.add_constant(X)

residual_model = sm.OLS(y, X).fit()
predictions = residual_model.predict(X)

print(residual_model.summary().tables[1])
print("R^2: %f" % residual_model.rsquared)

print("RSE: %f" % np.sqrt(residual_model.mse_resid))

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(xData, yData)

# Printing relevant values
print('(Intercept):', regr.intercept_)
print('Coefficients:', regr.coef_)
print('Residuals:', regr._residues)
print('R2 score', regr.score(xData, yData))

print("Mean squared error: %.2f" % np.mean((regr.predict(xData) - yData) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(xData, yData))

xData = np.array(xData)
# Plot outputs
dataPlot = plt.figure(1)
plt.scatter(xData, yData, color='black')
plt.plot(xData, regr.predict(xData), color='blue', linewidth=3)
plt.xlabel("LSTAT")
plt.ylabel("MEDV")

residuals = (yData - regr.predict(xData))
# Residuals vs Fitted
resVsFitted = plt.figure(2)
resVsFitted.axes[0] = sns.residplot(regr.predict(xData), residuals,
                                    lowess=True,
                                    scatter_kws={'alpha': 0.5},
                                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
resVsFitted.axes[0].set_title('Residuals vs Fitted')
resVsFitted.axes[0].set_xlabel('Fitted values')
resVsFitted.axes[0].set_ylabel('Residuals')

# QQ Plots
standardized_residuals = residuals / np.std(residuals)
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

# Scale location plot
model_std_residuals_abs_sqrt = np.sqrt(np.abs(standardized_residuals))
SL_plot = plt.figure(4)
plt.scatter(regr.predict(xData), model_std_residuals_abs_sqrt, alpha=0.5)
sns.regplot(regr.predict(xData), model_std_residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
SL_plot.axes[0].set_title('Scale-Location')
SL_plot.axes[0].set_xlabel('Fitted values')
SL_plot.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');
# annotations
abs_sq_norm_resid = np.flip(np.argsort(model_std_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
for i in abs_norm_resid_top_3:
    SL_plot.axes[0].annotate(i,
                             xy=(regr.predict(xData)[i],
                                 model_std_residuals_abs_sqrt[i]));

# Risidual plot
plt.figure(6)
plt.scatter(regr.predict(xData), regr.predict(xData) + yData)
# plt.scatter(regr.predict(xData), regr.predict(xData) - yData, c='b', s=40, alpha=0.5)
plt.show()
