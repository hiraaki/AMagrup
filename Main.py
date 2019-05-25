import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import math
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling


def ClusterIndicesNumpy(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]


def euclidiana(origem, fim):
    dif = []
    for i in range(len(origem)):
        point = []
        for j in range(len(origem[i])):
            aux = pow((origem[i][j] - fim[i][j]), 2)
            point.append(aux)
        dif.append(math.sqrt(sum(point)))
    return dif


def getgrouplabel(indexies, lista):
    lista_ = []
    for index in indexies:
        lista_.append((lista[index]))
    return lista_


data_frame = pd.read_csv("Maurício.csv")
x = data_frame .iloc[:, :2].values
y = data_frame .iloc[:, 2:].values
print(x[0])
print(y[0])

kmeans = KMeans(n_clusters=7, max_iter=300).fit(x)
cluster_map = pd.DataFrame()
cluster_map['data_index'] = data_frame.index.values
cluster_map['cluster'] = kmeans.labels_
print(ClusterIndicesNumpy(3, kmeans.labels_))
print(getgrouplabel(ClusterIndicesNumpy(3, kmeans.labels_), x))
print(euclidiana(getgrouplabel(ClusterIndicesNumpy(3, kmeans.labels_), x), getgrouplabel(ClusterIndicesNumpy(2, kmeans.labels_), x)))

#
# print(kmeans)
# print("\n", clusters)
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='random')
#     kmeans.fit(x)
#     print(i, kmeans.inertia_)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('O Metodo Elbow')
# plt.xlabel('Numero de Clusters')
# plt.ylabel('WSS') #within cluster sum of squares
# plt.show()


