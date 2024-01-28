import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.neighbors import NearestNeighbors

df =pd.read_csv("basketball_pass.csv")

# check the shape of dataset
df.shape

#Compute required parameters for DBSCAN clustering

# n_neighbors = 5 as kneighbors function returns distance of point to itself (i.e. first column will be zeros) 
nbrs = NearestNeighbors(n_neighbors=5).fit(df)
# Find the k-neighbors of a point
neigh_dist, neigh_ind = nbrs.kneighbors(df)
# sort the neighbor distances (lengths to points) in ascending order
# axis = 0 represents sort along first axis i.e. sort along row
sort_neigh_dist = np.sort(neigh_dist, axis=0)

#Graph to deduce the eps value 

k_dist = sort_neigh_dist[:, 4]
plt.plot(k_dist)
plt.axhline(y=2.5, linewidth=1, linestyle='dashed', color='k')
plt.ylabel("k-NN distance")
plt.xlabel("Sorted observations (4th NN)")
plt.figure()
plt.show()

#In the plot, the knee occurs at approximately 2.5 i.e. the points below 
#2.5 belong to a cluster and points above 2.5 are noise or outliers 
#(noise points will have higher kNN distance).

#minPts = 2 x 2 = 4
#e = 2.5

#Compute DBSCAN clustering

clusters = DBSCAN(eps=2.5, min_samples=4).fit(df)
# get cluster labels
clusters.labels_

# check unique clusters
set(clusters.labels_)
# -1 value represents noisy points could not assigned to any cluster

#Get each cluster size
Counter(clusters.labels_)

p = sns.scatterplot(data=df, x=" X (m/s2)", y=" Theta (deg)", hue=clusters.labels_, legend="full", palette="deep")
sns.move_legend(p, "upper right", bbox_to_anchor=(1.17, 1.2), title='Clusters')
plt.figure()
plt.show()
