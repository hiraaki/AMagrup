import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import math
import GroupTest as gt


data_frame = pd.read_csv("Maurício.csv")
x = data_frame.iloc[:, :2].values
y = data_frame.iloc[:, 2:].values
# gt.bestfitKmeans(data_frame)
coe = []
ent = []
sep = []
csi = []
for i in range(10):
    kmeans = KMeans(n_clusters=17, init='random', n_init=16, max_iter=110).fit(x)
    labels = kmeans.predict(x)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    grupos = gt.getgroups(17, labels, x)
    coe.append(gt.coesao(grupos))
    ent.append(gt.entropia(cluster_map))
    sep.append(gt.separabilidade(grupos))
    csi.append(gt.coe_Sirueta(grupos))
print("Resultado Kmeans")
print(np.mean(coe))
print(np.mean(ent))
print(np.mean(sep))
print(np.mean(csi))

coe.clear()
ent.clear()
sep.clear()
csi.clear()
while len(ent) != 10:
    dbscan = DBSCAN(eps=101.5, min_samples=3).fit(x)
    labels = dbscan.fit_predict(x)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    if (cluster_map.query('cluster != -1').count()[0] > 1) & (max(labels) >= 1):
        grupos = gt.getgroups(max(labels) + 1, labels, x)
        # print(cluster_map.query('cluster != -1'))
        # print(grupos)
        coe.append(gt.coesao(grupos))

        ent.append(gt.entropia(cluster_map))

        sep.append(gt.separabilidade(grupos))

        csi.append(gt.coe_Sirueta(grupos))
print("Resultado DBScan")
print(np.mean(coe))
print(np.mean(ent))
print(np.mean(sep))
print(np.mean(csi))

coe.clear()
ent.clear()
sep.clear()
csi.clear()
for i in range(10):
    agnes = AgglomerativeClustering(linkage="complete").fit(x)
    labels = agnes.labels_
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data_frame.index.values
    cluster_map['classe'] = data_frame.classe.values
    cluster_map['cluster'] = labels
    grupos = gt.getgroups(max(labels)+1, labels, x)
    coe.append(gt.coesao(grupos))
    ent.append(gt.entropia(cluster_map))
    sep.append(gt.separabilidade(grupos))
    csi.append(gt.coe_Sirueta(grupos))
print("Resultado Agnes")
print(np.mean(coe))
print(np.mean(ent))
print(np.mean(sep))
print(np.mean(csi))