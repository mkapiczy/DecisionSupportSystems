from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Create random variable with x,y coordinates
np.random.seed(123)
X = np.random.randn(50,2)

# Seperate data into two equally large groups 
X[0:25,0] = X[0:25,0]+4
X[0:25,1] = X[0:25,1]-4

# Plot the data
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Plot of original data points')

# Try k-means algorithm with different number of clusters and different initial starting points for each centroid. 
for k in 2,3:
    for n in 1,20:
        kmeans = KMeans(n_clusters=k, n_init=n, random_state=11).fit(X)
        print(kmeans.labels_)

        # Plot the cluster assignments
        plt.figure(figsize=(6,5))
        plt.scatter(X[:, 0], X[:, 1], s = 50, c = kmeans.labels_)
        plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker = '^',
            s = 100,
            color = 'black',
            label = 'Centers')
        plt.legend(loc = 'best')
        plt.xlabel('X0')
        plt.ylabel('X1')
        plt.title('Cluster assignment with: K=%i and n=%i' %(k,n))
        print('K=', k, 'n=', n, 'WCV=', kmeans.inertia_)
        
plt.show()