from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Create random variable with x,y coordinates
np.random.seed(321)
X = np.random.randn(50,2)

# Seperate data into two equally large groups 
X[0:25,0] = X[0:25,0]+4
X[0:25,1] = X[0:25,1]-4

# Plot the data
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], s=50)
ax.set_xlabel('X0')
ax.set_ylabel('X1')

kmeans = KMeans(n_clusters=2, random_state=123).fit(X)

print(kmeans.labels_)

plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], s = 50, c = kmeans.labels_, cmap = plt.cm.bwr)
plt.scatter(kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker = '^',
    s = 150,
    color = 'black',
    label = 'Centers')
plt.legend(loc = 'best')
plt.xlabel('X0')
plt.ylabel('X1')
plt.show()