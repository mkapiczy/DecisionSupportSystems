import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Smarket.csv', index_col=0, parse_dates = True)
summary = data.describe()

summary = summary.transpose()
print (summary)

corr = data.corr()
print (corr)

plt.figure(1)
plt.plot(data.Volume, 'bo', markersize = 3)
plt.xlabel("Day")
plt.ylabel("Volume (billions)")
plt.show()