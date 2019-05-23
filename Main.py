import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_frame = pd.read_csv("Maurício.csv")
x = data_frame .iloc[:, 0:2].values

# kmeans = KMeans(n_clusters=8, max_iter=300)
# clusters = kmeans.fit(x).predict(x)
#
# print(kmeans)
# print("\n", clusters)
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(X)
    print i,kmeans.inertia_
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()


